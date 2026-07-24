"""Issue #42: per-model ``n_seed_nonzero`` + ``n_independent_parameters``.

``jacobian_characterization.py`` records the Jacobian-side structure of every ODE
corpus model (``N``, ``rank``, ``density``, ``stiffness_ratio_*``) but used to record
nothing about model COMPOSITION, so the paper's representative-models table had to
carry its IC-not-zero and parameter-count columns as hand-curated constants.

:func:`seed_and_parameter_census` supplies both, and — because a parameter count is
only meaningful with the exclusion policy that produced it — emits the excluded names
and their reasons alongside.

These tests pin:

  * Unit (cheap, always run): the .net parameter parse (numeric literal vs expression,
    BNG's synthetic ``_rateLaw{N}``), the two exclusion reasons, the bookkeeping
    identity ``n_independent = n_parameters - len(excluded)``, and that a seed species
    whose parameter resolves to 0 is NOT counted while its parameter still IS.
  * Engine (needs BNG2.pl + perl): the exemplar values quoted in issue #42, generated
    end-to-end from the vendored BNGL — the numbers the paper's table asserts.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import _core
import pytest

_BNG_PARITY = Path(__file__).resolve().parent.parent / "bng_parity"
if str(_BNG_PARITY) not in sys.path:
    sys.path.insert(0, str(_BNG_PARITY))


def _load_jac():
    spec = importlib.util.spec_from_file_location(
        "jacobian_characterization", _BNG_PARITY / "jacobian_characterization.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


J = _load_jac()
MODELS = _BNG_PARITY / "models"

# A .net exercising every branch of the census in one file:
#   NA / V_ref  -- unit-conversion constants (excluded by name)
#   R_dim       -- derived (expression over Rtot)
#   _rateLaw1   -- BNG's synthetic compound-rate-law parameter (not a model parameter)
#   S0 = 0      -- a numeric parameter that IS independent but seeds a ZERO species
#   B, C        -- zero ICs (one via S0, one literal) that must not be counted
_NET = """\
begin parameters
    1 NA         6.02214076e23  # Constant
    2 V_ref      1e-12  # Constant
    3 kf         0.5  # Constant
    4 Rtot       100  # Constant
    5 R_dim      Rtot/2  # ConstantExpression
    6 S0         0  # Constant
    7 _rateLaw1  kf*Rtot  # ConstantExpression
end parameters
begin species
    1 A() Rtot
    2 B() S0
    3 C() 0
    4 D() R_dim
end species
begin reactions
    1 1 2 kf
end reactions
begin groups
    1 A_obs 1
end groups
"""


@pytest.fixture(scope="module")
def census(tmp_path_factory):
    from bngsim import Model

    net = tmp_path_factory.mktemp("census") / "census.net"
    net.write_text(_NET)
    return J.seed_and_parameter_census(Model.from_net(str(net)), _NET)


# --------------------------------------------------------------------------- #
# Unit: the .net parameter parse.
# --------------------------------------------------------------------------- #
def test_parse_net_parameters_splits_numeric_literals_from_expressions():
    parsed = J._parse_net_parameters(_NET)
    assert [name for name, _, _ in parsed] == [
        "NA",
        "V_ref",
        "kf",
        "Rtot",
        "R_dim",
        "S0",
        "_rateLaw1",
    ]
    assert dict((name, derived) for name, _, derived in parsed) == {
        "NA": False,
        "V_ref": False,
        "kf": False,
        "Rtot": False,
        "R_dim": True,  # Rtot/2 -- not a numeric literal
        "S0": False,  # 0 is numeric-valued, so still an independent knob
        "_rateLaw1": True,
    }


def test_parse_net_parameters_ignores_everything_outside_the_block():
    """Species/reaction/group lines never leak in, and a file with no block is empty."""
    assert J._parse_net_parameters("begin species\n    1 A() 1\nend species\n") == []


# --------------------------------------------------------------------------- #
# Unit: the census itself.
# --------------------------------------------------------------------------- #
def test_seed_nonzero_counts_resolved_initial_values(census):
    """A(Rtot)=100 and D(R_dim)=50 count; B(S0)=0 and C=0 do not. Counting seed LINES
    would give 4 -- the bug this field exists to avoid."""
    assert census["n_seed_nonzero"] == 2


def test_synthetic_ratelaw_parameters_are_not_model_parameters(census):
    """``_rateLaw1`` is BNG's own symbol for a compound rate law: it is in neither the
    declared total nor the exclusion list (which would misreport it as curation)."""
    assert census["n_parameters"] == 6
    assert "_rateLaw1" not in census["excluded_parameters"]


def test_exclusions_are_named_with_their_reason(census):
    assert census["excluded_parameters"] == ["NA", "V_ref", "R_dim"]
    assert census["excluded_parameter_reasons"] == {
        "NA": "unit_conversion",
        "V_ref": "unit_conversion",
        "R_dim": "derived",
    }


def test_independent_count_is_reproducible_from_the_emitted_names(census):
    """The whole point of emitting the names: the count is auditable arithmetic, not a
    hidden judgment. kf, Rtot, S0 survive."""
    assert census["n_independent_parameters"] == 3
    assert census["n_independent_parameters"] == census["n_parameters"] - len(
        census["excluded_parameters"]
    )


def test_avogadro_spelling_variants_are_recognized(tmp_path):
    """Models scale Avogadro's number to their own units (6.02e8 /um^3 in the RuleHub
    ``LR``/``LV`` tutorials, 6.022e23 /mol elsewhere) and spell it several ways."""
    from bngsim import Model

    net = _NET.replace("1 NA         6.02214076e23", "1 NaV        6.02e8")
    p = tmp_path / "avogadro.net"
    p.write_text(net)
    c = J.seed_and_parameter_census(Model.from_net(str(p)), net)
    assert c["excluded_parameter_reasons"]["NaV"] == "unit_conversion"


def test_avogadro_name_match_is_guarded_by_magnitude_and_case():
    """``nA 5`` is ``race.bngl``'s Erlang step count, not Avogadro's number — a
    case-insensitive name match would silently delete a real model parameter. A
    correctly-spelled ``NA`` holding a small value is likewise not Avogadro."""
    assert not J._is_unit_conversion("nA", "5")
    assert not J._is_unit_conversion("NA", "5")
    assert J._is_unit_conversion("NA", "6.02214076e23")
    assert J._is_unit_conversion("Na", "6.022e23")  # CaOscillate_Sat_2.bngl spelling


def test_geometry_names_rest_on_the_name_alone():
    """A reference volume has no distinguishing magnitude (1e-12 L/cell, 1000 um^3),
    so these match by name only -- and only in their documented spellings."""
    assert J._is_unit_conversion("V_ref", "1e-12")
    assert J._is_unit_conversion("Vcell", "1000")
    assert not J._is_unit_conversion("V", "1000")
    assert not J._is_unit_conversion("v_ref", "1e-12")


# --------------------------------------------------------------------------- #
# Engine: the exemplar values issue #42 quotes for the paper's table.
# --------------------------------------------------------------------------- #
_BNG = _core.resolve_bng()
engine = pytest.mark.skipif(not _BNG.ok, reason=_BNG.why_not())

# (model_id, N, n_seed_nonzero, n_independent_parameters). Two of the six exemplars:
# Lang_2024 is the parameter-valued-IC case (73 seed lines, 8 resolve to 0) with no
# exclusions at all, Barua_2007 the case that exercises BOTH exclusion reasons.
# The other four (Kocieniewski 10, Blinov 43, Barua_2013 25, fceri_fyn 31) need
# minutes of network generation apiece and are left to a full corpus run.
_EXEMPLARS = [
    ("fast/rulehub/Published/Lang2024/Lang_2024.bngl", 73, 65, 205),
    ("slow/rulehub/Published/Barua2007/Barua_2007.bngl", 149, 2, 22),
]


@engine
@pytest.mark.parametrize(("model_id", "n", "ic", "par"), _EXEMPLARS)
def test_exemplar_composition_counts(model_id, n, ic, par):
    row = J.characterize_model(model_id, {}, str(_BNG.bng2_pl), timeout=600)
    assert str(row.get("status", "")).startswith("ok"), row
    assert (row["N"], row["n_seed_nonzero"], row["n_independent_parameters"]) == (n, ic, par)
    assert row["n_independent_parameters"] == row["n_parameters"] - len(row["excluded_parameters"])


@engine
def test_barua_2007_exclusions_are_the_curated_ones():
    """The five names that take the mechanical 24 down to the caption's 22: two unit
    constants plus three parameters defined from others."""
    row = J.characterize_model(_EXEMPLARS[1][0], {}, str(_BNG.bng2_pl), timeout=600)
    assert row["excluded_parameter_reasons"] == {
        "NA": "unit_conversion",
        "V_ref": "unit_conversion",
        "koff_CSH2": "derived",
        "koff_NSH2": "derived",
        "R_dim": "derived",
    }
