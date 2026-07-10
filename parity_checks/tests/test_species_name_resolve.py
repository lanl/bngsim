"""GH #182: canonicalize a setConcentration pattern to the .net species name.

``bngsim.Model.set_concentration`` resolves a species by its EXACT canonical string, so
a BNGL pattern differing only by component order or ``@c:``/``@c::`` compartment syntax
forces the multi-phase replay onto the single-segment fallback. ``resolve_species_name``
maps such a pattern to the canonical name — but ONLY on an unambiguous, bond-free,
concrete match, so an ambiguous or malformed pattern still falls back rather than
silently substituting the wrong species. These cases mirror the 3 corpus models in #182.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bng_parity"))

import _bng_common as bc  # noqa: E402


def test_component_reorder_resolves_uniquely():
    # modified_THEM_v1_1: BSA_THEM(DNP~hidden,DNP~exposed) -> canonical (DNP~exposed,DNP~hidden)
    canonical = [
        "BSA_THEM(DNP~hidden,DNP~hidden)",
        "BSA_THEM(DNP~exposed,DNP~hidden)",
        "BSA_THEM(DNP~exposed,DNP~exposed)",
    ]
    assert (
        bc.resolve_species_name("BSA_THEM(DNP~hidden,DNP~exposed)", canonical)
        == "BSA_THEM(DNP~exposed,DNP~hidden)"
    )


def test_compartment_syntax_resolves():
    # Lang_2024: @cell:TP53(DBD,Ser15~p) -> canonical @cell::TP53(DBD,Ser15~p)
    canonical = ["@cell::TP53(DBD,Ser15~p)", "@cell::CDKN1A_promoter(TP53)"]
    assert (
        bc.resolve_species_name("@cell:TP53(DBD,Ser15~p)", canonical) == "@cell::TP53(DBD,Ser15~p)"
    )


def test_malformed_pattern_falls_back_unchanged():
    # HarmonicOscillator_redo_v1: IGF1(ds,hs,label~cold  (unbalanced paren — a model typo).
    # Must NOT be "repaired" — return unchanged so the exact-name path raises -> fallback.
    canonical = ["IGF1(ds,hs,label~hot)", "IGF1(ds,hs,label~cold)"]
    pat = "IGF1(ds,hs,label~cold"
    assert bc.resolve_species_name(pat, canonical) == pat


def test_exact_name_is_untouched():
    canonical = ["BSA_THEM(DNP~exposed,DNP~exposed)"]
    assert (
        bc.resolve_species_name("BSA_THEM(DNP~exposed,DNP~exposed)", canonical)
        == "BSA_THEM(DNP~exposed,DNP~exposed)"
    )


def test_no_match_falls_back_unchanged():
    canonical = ["A(x)", "A(y)"]
    assert bc.resolve_species_name("B(z)", canonical) == "B(z)"


def test_ambiguous_match_falls_back_never_guesses():
    # Two distinct canonical species that collapse to the same bond-free key must NOT be
    # resolved — returning unchanged is the safe (fallback) behavior. (Constructed: a real
    # .net never lists two identical canonical species, but the resolver must be defensive.)
    canonical = ["A(p,q)", "A(q,p)"]
    assert bc.resolve_species_name("A(p,q)", canonical) == "A(p,q)"  # exact wins first
    assert (
        bc.resolve_species_name("A(q , p)", canonical) == "A(q , p)"
    )  # 2 key-matches -> fallback


def test_bonded_complex_pattern_is_not_guessed():
    # A complex/bonded pattern (contains '.' or '!') is non-concrete for this purpose;
    # return unchanged so we never guess a complex's canonical ordering.
    canonical = ["A(b!1).B(a!1)"]
    assert bc.resolve_species_name("B(a!1).A(b!1)", canonical) == "B(a!1).A(b!1)"
