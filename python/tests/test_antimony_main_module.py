"""The Antimony loaders build the main module, not the last one (issue #1508).

``Model.from_antimony_string``, ``Model.from_antimony`` and ``Model.load`` on a
``.ant`` path built ``getModuleNames()[-1]``, the last module defined. That is
the main module only when nothing stands outside a model and no other model is
marked ``*``. With top-level reactions beside a helper ``model``, a
``model *main()`` defined before another, or a module instantiated at top level
with an override, bngsim simulated the other module, with no word.

They now build the module libantimony's ``getMainModuleName()`` names, which
is the module RoadRunner (through tellurium's ``loada``) simulates. In every case
below the main module decays S1 from 10 at k = 0.5, so S1(2) = 10*exp(-1).
Where a stray top-level statement makes the top level the main module with
nothing in it that changes, the last module defined is loaded as before.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest

pytest.importorskip("antimony")

import bngsim  # noqa: E402
import numpy as np  # noqa: E402

S1_AT_2 = 10 * math.exp(-0.5 * 2)
MAIN = "J: S1 => S2; k*S1\n  S1 = 10; S2 = 0; k = 0.5"
HELPER = "model helper()\n  J: X => Y; kx*X\n  X = 1; Y = 0; kx = 9\nend"


def _first_at_2(model: bngsim.Model) -> float:
    result = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 2.0), n_points=3, rtol=1e-10, atol=1e-12
    )
    return float(np.asarray(result.species)[-1, 0])


# ── The main module, where the last one was loaded ───────────────────────────

FIXED = {
    # libantimony: modules ['__main', 'helper'], main '__main'. Main loaded X, Y.
    "top-level-reactions-and-a-helper": (f"{MAIN}\n{HELPER}", ["S1", "S2"]),
    # modules ['__main', 'main', 'helper'], main 'main'.
    "a-starred-model-before-a-helper": (f"model *main()\n  {MAIN}\nend\n{HELPER}", ["S1", "S2"]),
    # modules ['__main', 'sub'], main '__main', which holds the override. Main
    # loaded sub at its own k = 0.1: 8.187307.
    "an-instance-with-an-override": (
        "model sub()\n  J: S1 => S2; k*S1\n  S1 = 10; S2 = 0; k = 0.1\nend\nA: sub();\nA.k = 0.5",
        ["A_S1", "A_S2"],
    ),
}


@pytest.mark.parametrize("case", sorted(FIXED))
def test_the_main_module_is_built(case):
    text, species = FIXED[case]
    model = bngsim.Model.from_antimony_string(text)
    assert list(model.species_names) == species
    assert _first_at_2(model) == pytest.approx(S1_AT_2, rel=1e-6)


def test_a_rate_rule_at_the_top_level_counts_as_the_model():
    """A main module with a rate rule and no reaction is a model: the top
    level is loaded, not the helper after it."""
    model = bngsim.Model.from_antimony_string(f"S1' = -k*S1\nS1 = 10; k = 0.5\n{HELPER}")
    assert _first_at_2(model) == pytest.approx(S1_AT_2, rel=1e-6)
    assert "X" not in model.species_names and "X" not in model.param_names


@pytest.mark.parametrize(
    "top",
    [
        # An event and no reaction or rule.
        "species S1 = 10\nat (time > 1): S1 = 0",
        # An assignment rule and no reaction or event.
        "species S1 = 10\nr := 2*time",
    ],
)
def test_an_event_or_an_assignment_rule_counts_as_the_model(top):
    """Each changes something over time, so the top level is the model."""
    model = bngsim.Model.from_antimony_string(f"{top}\n{HELPER}")
    assert "S1" in model.species_names and "X" not in model.species_names


def test_every_entry_point_builds_it(tmp_path):
    """``from_antimony``, ``Model.load`` and an ``EvaluationSpec`` read the
    same text through the same loader."""
    text = f"{MAIN}\n{HELPER}"
    path = tmp_path / "m.ant"
    path.write_text(text)
    for model in (
        bngsim.Model.from_antimony(str(path)),
        bngsim.Model.load(str(path)),
        bngsim.EvaluationSpec(model_source=text, model_format="antimony_string").build_model(),
        bngsim.EvaluationSpec(model_source=str(path), model_format="antimony").build_model(),
    ):
        assert list(model.species_names) == ["S1", "S2"]


def test_a_file_that_imports_a_library_builds_its_own_top_level(tmp_path):
    """A top-level file that imports a module library and instantiates one."""
    (tmp_path / "lib.ant").write_text(
        f"model sub()\n  J: S1 => S2; k*S1\n  S1 = 10; S2 = 0; k = 0.1\nend\n{HELPER}\n"
    )
    main = tmp_path / "main.ant"
    main.write_text('import "lib.ant"\nA: sub();\nA.k = 0.5\n')
    model = bngsim.Model.load(str(main))
    assert list(model.species_names) == ["A_S1", "A_S2"]
    assert _first_at_2(model) == pytest.approx(S1_AT_2, rel=1e-6)


# ── Read as before ───────────────────────────────────────────────────────────

AS_BEFORE = {
    "one-model": (f"model m()\n  {MAIN}\nend", ["S1", "S2"]),
    "top-level-only": (MAIN, ["S1", "S2"]),
    "two-models-the-model-last": (f"{HELPER}\nmodel m()\n  {MAIN}\nend", ["S1", "S2"]),
    # With no star and nothing at the top level, the main module is the last
    # one defined, as Antimony defines it: here the helper.
    "two-models-the-helper-last": (f"model m()\n  {MAIN}\nend\n{HELPER}", ["X", "Y"]),
}


@pytest.mark.parametrize("case", sorted(AS_BEFORE))
def test_a_text_whose_main_module_is_its_last_reads_as_before(case):
    """Control: loaded on main too."""
    text, species = AS_BEFORE[case]
    with warnings.catch_warnings():
        warnings.simplefilter("error", bngsim.ConversionWarning)
        model = bngsim.Model.from_antimony_string(text)
    assert list(model.species_names) == species


def test_a_unit_line_before_the_model_loads_the_model_silently():
    """Control: a top level of unit definitions alone is the main module with
    nothing in it; the model after it is loaded, as on main, with no warning."""
    text = f"unit nM = 1e-9 mole / litre\nmodel m()\n  {MAIN}\nend"
    with warnings.catch_warnings():
        warnings.simplefilter("error", bngsim.ConversionWarning)
        model = bngsim.Model.from_antimony_string(text)
    assert _first_at_2(model) == pytest.approx(S1_AT_2, rel=1e-6)


@pytest.mark.parametrize(
    ("stray", "name"),
    [("species Z = 1", "Z"), ("const q = 3", "q")],
)
def test_a_stray_statement_s_symbol_is_named_as_left_out(stray, name):
    """The model loads as on main; the top-level symbol it drops, which main
    dropped with no word, is named."""
    text = f"model m()\n  {MAIN}\nend\n{stray}"
    with pytest.warns(bngsim.ConversionWarning, match=rf"top level's {name} left out") as caught:
        model = bngsim.Model.from_antimony_string(text)
    assert "model *m()" in str(caught[0].message)
    assert _first_at_2(model) == pytest.approx(S1_AT_2, rel=1e-6)


def test_the_roadrunner_oracle_reads_the_main_module_too(tmp_path):
    """``harness/common.py``'s ``_ant_to_sbml``, the RoadRunner reference side
    of the comparisons, took the last module as bngsim did, so the two agreed
    on the wrong model and no comparison could catch it."""
    import importlib.util

    source = Path(__file__).resolve().parents[2] / "harness" / "common.py"
    spec = importlib.util.spec_from_file_location("_harness_common_1508", source)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    path = tmp_path / "m.ant"
    path.write_text(f"{MAIN}\n{HELPER}\n")
    sbml = harness._ant_to_sbml(str(path))
    assert 'id="S1"' in sbml and 'id="X"' not in sbml
