"""Tests for bngsim.convert.net_to_sbml — .net→SBML exporter (GH #216).

The reverse of sbml2net (#215). Covers the round-trip (structural + ODE-RHS)
equivalence over a tracked ``.net`` set, the ExprTk→MathML translator, the MM
(tQSSA) and BNG zero-arg-call cases, the capability boundary (events dropped;
``tfun`` table functions and non-unit compartment volumes refused under strict),
the faithful half (unit-volume SBML — including amount-valued species —
round-trips through ``write_sbml``), the L2 round-trip-identity seed that
unblocks the #217 validation framework, and the CLI entry point.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim.convert import (
    ConversionReport,
    net_to_sbml,
    sbml_to_net,
    validate_roundtrip,
    write_sbml,
)
from bngsim.convert._sbml_writer import (
    _MATHML_RESERVED_SIDS,
    _exprtk_to_ast,
    _sanitize_sid,
    sbml_capability_report,
)

pytestmark = pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SBML conversion requires libsbml")

_BNGSIM = Path(__file__).resolve().parents[2]
_DATA = _BNGSIM / "tests" / "data"
_EVENTS = _BNGSIM / "benchmarks" / "sbml_events"


@pytest.fixture
def data_dir() -> Path:
    assert _DATA.is_dir(), f"Test data directory not found: {_DATA}"
    return _DATA


# Tracked, network-clean .net models that convert faithfully (unit volume).
_CLEAN_NETS = [
    "simple_decay.net",
    "case_sensitivity.net",
    "derived_rate_const.net",
    "derived_quotient.net",
    "two_species_reversible.net",
    "t_as_observable.net",
    "fixed_species.net",
    "preequil_prod_deg.net",
    "time_dependent_func.net",
    "mm_tqssa.net",
    "obs_zero_arg_call.net",
]


def _rhs_max_delta(a: bngsim.Model, b: bngsim.Model) -> float:
    y0 = np.asarray(a.get_state(), dtype=float)
    worst = 0.0
    for t, y in ((0.0, y0), (1.0, y0 * 1.37 + 0.05)):
        ra = np.asarray(a._core._eval_rhs(t, y), dtype=float)
        rb = np.asarray(b._core._eval_rhs(t, y), dtype=float)
        scale = max(float(np.abs(ra).max(initial=0.0)), 1.0)
        worst = max(worst, float(np.max(np.abs(ra - rb)) / scale))
    return worst


# ─── Round-trip (structural + RHS) over the tracked clean set ──────────────


@pytest.mark.parametrize("name", _CLEAN_NETS)
def test_roundtrip_ok(name: str, data_dir: Path, tmp_path: Path) -> None:
    src = data_dir / name
    if not src.is_file():
        pytest.skip(f"tracked .net not present: {src}")
    out = tmp_path / (src.stem + ".xml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = net_to_sbml(src, out, validate="L1", strict=True)

    assert isinstance(report, ConversionReport)
    assert out.is_file()
    assert report.structural is not None
    assert report.structural.passed, report.structural.summary()
    assert report.ok
    # The numerical equivalence is exact for these models.
    assert report.max_rhs_delta is not None and report.max_rhs_delta <= 1e-9


@pytest.mark.parametrize("name", _CLEAN_NETS)
def test_output_loads_via_from_sbml(name: str, data_dir: Path, tmp_path: Path) -> None:
    src = data_dir / name
    if not src.is_file():
        pytest.skip(f"tracked .net not present: {src}")
    out = tmp_path / (src.stem + ".xml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_sbml(src, out, validate=None, strict=True)
        src_model = bngsim.Model.from_net(src)
        sbml_model = bngsim.Model.from_sbml(out)

    assert sbml_model.n_species == src_model.n_species
    assert sbml_model.n_reactions == src_model.n_reactions
    assert _rhs_max_delta(src_model, sbml_model) <= 1e-9


# ─── Michaelis–Menten (tQSSA) reconstruction ───────────────────────────────


def test_mm_tqssa_rhs_exact(data_dir: Path, tmp_path: Path) -> None:
    src = data_dir / "mm_tqssa.net"
    if not src.is_file():
        pytest.skip("mm_tqssa.net not present")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        src_model = bngsim.Model.from_net(src)
        write_sbml(src_model, tmp_path / "mm.xml", strict=True)
        sbml_model = bngsim.Model.from_sbml(tmp_path / "mm.xml")
    # tQSSA is a nonlinear closed form; exact reconstruction means exact RHS.
    assert _rhs_max_delta(src_model, sbml_model) <= 1e-9


# ─── ExprTk→MathML translator unit checks ──────────────────────────────────


def test_translator_constructs() -> None:
    import libsbml

    cases = {
        "(k3*S1)/(K3+S2)": "k3 * S1 / (K3 + S2)",
        "if(time()>5, a, b)": "piecewise(a, time > 5, b)",
        "if(A>0, if(B>0, 1, 2), 3)": "piecewise(piecewise(1, B > 0, 2), A > 0, 3)",
        "_pi*r^2": "pi * r^2",
    }
    for expr, want in cases.items():
        ast = _exprtk_to_ast(expr, where="t")
        assert libsbml.formulaToL3String(ast) == want, expr


def test_translator_infix_and_or() -> None:
    """ExprTk infix ``and``/``or`` are keyword operators; ``parseL3Formula`` only
    accepts ``&&``/``||`` (the bareword is a syntax error). The translator must
    rewrite them — the dominant shape in time-gated forcing functions — while
    leaving symbols that merely *contain* the letters (``ligand``, ``factor``)
    untouched."""
    import libsbml

    ast = _exprtk_to_ast("if((time()>=50) and (time()<=100), 1, 0)", where="t")
    assert libsbml.formulaToL3String(ast) == "piecewise(1, (time >= 50) && (time <= 100), 0)"

    ast = _exprtk_to_ast("if((a<=1) or (b>=2), 1, 0)", where="t")
    assert libsbml.formulaToL3String(ast) == "piecewise(1, (a <= 1) || (b >= 2), 0)"

    # A symbol containing "and"/"or" as a substring is not an operator.
    ast = _exprtk_to_ast("ligand * factor", where="t")
    assert libsbml.formulaToL3String(ast) == "ligand * factor"


def test_translator_log_is_natural() -> None:
    """ExprTk ``log`` is natural log; SBML ``log`` is base-10. The translator
    must map ``log`` → ``ln`` and keep ``log10`` as ``log10``."""
    import libsbml

    ast = _exprtk_to_ast("log(x) + log10(y)", where="t")
    assert libsbml.formulaToL3String(ast) == "ln(x) + log10(y)"


def test_translator_log2_and_log1p() -> None:
    """ExprTk ``log2``/``log1p`` have no MathML primitive but reduce to ``ln``
    exactly: ``log2(x) = ln(x)/ln(2)``, ``log1p(x) = ln(1+x)``. The translator
    rewrites them (incl. nested-parenthesized args) rather than refusing."""
    import libsbml

    ast = _exprtk_to_ast("log2((a*b)/c)", where="t")
    assert libsbml.formulaToL3String(ast) == "ln(a * b / c) / ln(2)"

    ast = _exprtk_to_ast("log1p(x*2)", where="t")
    assert libsbml.formulaToL3String(ast) == "ln(1 + x * 2)"

    # Nested log2 inside log2 — both rewritten.
    ast = _exprtk_to_ast("log2(log2(x))", where="t")
    assert libsbml.formulaToL3String(ast) == "ln(ln(x) / ln(2)) / ln(2)"


def test_translator_pi_spelled_symbol_not_constant() -> None:
    """``parseL3Formula`` reads ``pi`` (case-insensitively: ``pI``/``PI``/``Pi``)
    as the π constant. A model symbol so named must survive as a name reference,
    not silently collapse to ``<pi/>`` (a wrong RHS). Genuine π (ExprTk ``_pi``)
    must still become the constant."""
    import libsbml

    # pi-spelled model symbols stay names.
    ast = _exprtk_to_ast("3*pI + Inorm", where="t")
    assert libsbml.formulaToL3String(ast) == "3 * pI + Inorm"
    ast = _exprtk_to_ast("PI + Pi + pi", where="t")
    assert libsbml.formulaToL3String(ast) == "PI + Pi + pi"
    for tok in ("pI", "PI", "Pi", "pi"):
        node = _exprtk_to_ast(tok, where="t")
        assert node.getType() == libsbml.AST_NAME

    # Genuine π (ExprTk `_pi`) is still the constant and serializes as <pi/>.
    ast = _exprtk_to_ast("2*_pi", where="t")
    assert "<pi/>" in libsbml.writeMathMLToString(ast)
    assert libsbml.formulaToL3String(ast) == "2 * pi"


def test_translator_zero_arg_observable_call() -> None:
    """An observable referenced as ``Atot()`` collapses to the bareword."""
    import libsbml

    ast = _exprtk_to_ast("k1*Atot()", where="t", scalar_names=frozenset({"k1", "Atot"}))
    assert libsbml.formulaToL3String(ast) == "k1 * Atot"


# ─── Reserved MathML-constant SIds (GH #8) ──────────────────────────────────


def test_sanitize_sid_avoids_mathml_constant_names() -> None:
    """A sanitized ``SId`` must never equal a MathML reserved constant bareword
    (``pi``, ``time``, ``avogadro``, …) — ``parseL3Formula`` reads those
    case-insensitively as ``<pi/>`` / the time csymbol / etc., silently
    mis-lowering any symbol referenced by that ``SId``. ``_sanitize_sid`` bumps
    them (``pi`` → ``pi_2``) so no emitted formula token can collapse (GH #8)."""
    import libsbml

    for word in _MATHML_RESERVED_SIDS:
        for cand in (word, word.upper(), word.capitalize()):
            sid = _sanitize_sid(cand, set(), fallback="s")
            assert sid.lower() not in _MATHML_RESERVED_SIDS, (cand, sid)
            # the invariant that matters: the SId parses as a plain name reference
            node = libsbml.parseL3Formula(sid)
            assert node is not None and node.getType() == libsbml.AST_NAME, (cand, sid)

    # Ordinary names (even ones that merely contain a reserved word) are untouched.
    for ok in ("k1", "Atot", "rho", "piston", "time_course", "runtime"):
        assert _sanitize_sid(ok, set(), fallback="s") == ok

    # Reserved + collision compose: two ``pi``-sanitizing symbols stay distinct
    # and neither collapses.
    used: set[str] = set()
    a = _sanitize_sid("pi()", used, fallback="s")
    b = _sanitize_sid("pi[]", used, fallback="s")
    assert a != b and a.lower() != "pi" and b.lower() != "pi"


# A minimal model reproducing the issue: a population species literally named
# ``pi()`` (``A`` decays into it, so it grows) and a Species-observable ``Ptot``
# over it. Before the fix the species took ``SId`` ``pi`` and ``Ptot``'s
# assignment rule serialized to the π *constant* ``<pi/>`` (≡ 3.14159), not a
# reference to the growing population.
_PI_SPECIES_NET = """\
begin parameters
    1 k1  0.5
end parameters
begin species
    1 A() 100
    2 pi() 0
end species
begin reactions
    1 1 2 k1 #_R1
end reactions
begin groups
    1 Atot  1
    2 Ptot  2
end groups
"""


def test_pi_named_species_observable_not_constant(tmp_path: Path) -> None:
    """A species/observable named ``pi`` round-trips to the population count, not
    the π constant, with no stray ``<pi/>`` in the emitted SBML (GH #8)."""
    import libsbml

    src = tmp_path / "pi_species.net"
    src.write_text(_PI_SPECIES_NET)
    out = tmp_path / "pi_species.xml"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_sbml(src, out, validate=None, strict=True)
    text = out.read_text()

    # No MathML π constant anywhere — the whole point of the bug.
    assert "<pi/>" not in text

    # The species named ``pi()`` got a non-``pi`` SId, and ``Ptot``'s assignment
    # rule is a plain <ci> reference to it — not the AST_CONSTANT_PI node.
    doc = libsbml.readSBMLFromString(text)
    m = doc.getModel()
    pi_sp = next(
        m.getSpecies(i) for i in range(m.getNumSpecies()) if m.getSpecies(i).getName() == "pi()"
    )
    assert pi_sp.getId().lower() != "pi"
    # observable parameter keeps its natural SId ``Ptot``; its rule is a <ci> ref
    assert m.getParameter("Ptot") is not None
    rule = m.getRuleByVariable("Ptot")
    assert rule is not None
    math = rule.getMath()
    assert math.getType() == libsbml.AST_NAME
    assert math.getName() == pi_sp.getId()


def test_pi_named_species_numeric_roundtrip(tmp_path: Path) -> None:
    """The reloaded model evaluates ``Ptot`` as the growing population, matching
    the source ``.net`` exactly — never pinned to π ≈ 3.14159 (GH #8)."""
    src = tmp_path / "pi_species.net"
    src.write_text(_PI_SPECIES_NET)
    out = tmp_path / "pi_species.xml"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_sbml(src, out, validate=None, strict=True)
        net_model = bngsim.Model.from_net(src)
        sbml_model = bngsim.Model.from_sbml(out)

    rn = bngsim.Simulator(net_model, method="ode").run(t_span=(0, 5), n_points=6)
    rs = bngsim.Simulator(sbml_model, method="ode").run(t_span=(0, 5), n_points=6)
    # The .net carries ``Ptot`` as an observable; the SBML round-trip re-expresses
    # it as an assignment-rule parameter (reloaded as an expression).
    ptot_net = np.asarray(rn.observables["Ptot"])
    ptot_sbml = np.asarray(rs.expressions["Ptot"])
    assert ptot_sbml[-1] > 1.0  # grew — not frozen at the π constant
    assert not np.allclose(ptot_sbml, np.pi, atol=1e-3)
    np.testing.assert_allclose(ptot_net, ptot_sbml, rtol=1e-9, atol=1e-9)


def test_translator_refuses_tfun() -> None:
    with pytest.raises(bngsim.ConversionError):
        _exprtk_to_ast("tfun('drive.tfun', time())", where="t")


# ─── Capability boundary ───────────────────────────────────────────────────


def test_tfun_refused_strict(data_dir: Path, tmp_path: Path) -> None:
    src = data_dir / "tfun_time_indexed.net"
    if not src.is_file():
        pytest.skip("tfun_time_indexed.net not present")
    with pytest.raises(bngsim.ConversionError) as exc:
        net_to_sbml(src, tmp_path / "t.xml", strict=True)
    msg = str(exc.value)
    assert "tfun" in msg or "table" in msg
    assert "strict=False" in msg or "--allow-lossy" in msg


def test_rint_refused_strict(data_dir: Path, tmp_path: Path) -> None:
    src = data_dir / "expr_sens_unsupported.net"
    if not src.is_file():
        pytest.skip("expr_sens_unsupported.net not present")
    with pytest.raises(bngsim.ConversionError):
        net_to_sbml(src, tmp_path / "u.xml", strict=True)


# ─── Volume-faithful kinetics: static non-unit volumes round-trip (#216) ───


# Static non-unit-volume SBML models present in the benchmark suite, spanning
# the distinguishing shapes: two-volume single-compartment kinetics (BIOMD1),
# genuine cross-compartment reactions (BIOMD88/342 — per_species_volume_scaling),
# amount-valued species at non-unit volume (MODEL1710030000), many distinct
# static volumes (MODEL1805140002, 7 of them), and a PBPK model whose 16 organ
# compartments have *constant* assignment-rule volumes (BIOMD1039,
# `organ = bodyweight · fraction` from constant parameters). These carry no
# time-varying-volume rescale, so both dynamics and reporting round-trip.
_NONUNIT_VOLUME_MODELS = [
    "BIOMD0000000001.xml",
    "BIOMD0000000088.xml",
    "BIOMD0000000342.xml",
    "MODEL1710030000.xml",
    "MODEL1805140002.xml",
    "BIOMD0000001039.xml",
]


@pytest.mark.parametrize("name", _NONUNIT_VOLUME_MODELS)
def test_nonunit_volume_roundtrips_faithfully(name: str, tmp_path: Path) -> None:
    """Static non-unit compartment volumes are now volume-faithful (#216).

    The writer scales each kinetic law by its reaction's compartment volume so
    SBML's d[conc]/dt = kineticLaw/V semantics reconstruct the engine's stored
    concentration rate. So these are NOT refused under strict, carry no
    ``volume``/``mis-scale`` lossy note, and round-trip RHS-exact — including
    cross-compartment reactions (per-species volume scaling) and amount-valued
    species at non-unit volume."""
    src = _EVENTS / name
    if not src.is_file():
        pytest.skip(f"{name} not present")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        model = bngsim.Model.from_sbml(src)
        # carries a non-unit volume in the first place
        data = model._core.codegen_data()
        vols = {float(s.get("volume_factor", 1.0)) for s in data["species"]}
        assert vols != {1.0}, f"{name} has only unit volume; not a useful case"
        rep = sbml_capability_report(model)
        assert not any("volume" in note or "mis-scale" in note for note in rep["lossy"]), rep[
            "lossy"
        ]
        write_sbml(model, tmp_path / "rt.xml", strict=True)  # must not raise
        back = bngsim.Model.from_sbml(tmp_path / "rt.xml")
    assert _rhs_max_delta(model, back) <= 1e-9


def test_cross_compartment_uniform_propensity_refused() -> None:
    """A uniform-propensity (``per_species_volume_scaling=False``) reaction whose
    dynamic species span more than one compartment volume is unrepresentable as
    a single SBML kinetic law and fails loud. No benchmark model exercises this
    (the loader routes genuine cross-compartment reactions through per-species
    scaling), so probe the volume-factor resolver directly."""
    from bngsim.convert._sbml_writer import _reaction_volume_factor

    rxn = {
        "per_species_volume_scaling": False,
        "reactants": [0],
        "products": [1],
    }
    species = [
        {"volume_factor": 1.0, "fixed": False},
        {"volume_factor": 0.5, "fixed": False},
    ]
    with pytest.raises(bngsim.ConversionError, match="different volume"):
        _reaction_volume_factor(rxn, species, idx=0)
    # the same shape with the second species *fixed* (no derivative) is fine,
    # scaling by the lone dynamic species' volume
    species[1]["fixed"] = True
    assert _reaction_volume_factor(rxn, species, idx=0) == 1.0


# A self-contained growing-compartment model: cell volume V(t) = 2 + 0.5·t via a
# rate rule; species S is a fixed amount in it, so its reported concentration
# tracks A/V(t). The kinetics (here, none) round-trip, but a static SBML document
# cannot carry the moving volume — the reported [S] would freeze at A/V_static.
_GROWING_VOLUME_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="grow">
    <listOfCompartments>
      <compartment id="cell" spatialDimensions="3" size="2" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="cell" initialAmount="10"
               hasOnlySubstanceUnits="true" boundaryCondition="true" constant="true"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="g" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="cell">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>g</ci></math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>
"""


def test_time_varying_volume_refused_strict(tmp_path: Path) -> None:
    """A time-varying (rate-rule-driven) compartment volume is refused under
    strict — static SBML cannot carry the moving volume, so a species whose
    reported concentration tracks it would round-trip mis-reported. Regression
    guard: the static-volume kinetics fix must NOT silently un-refuse this."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        model = bngsim.Model.from_sbml_string(_GROWING_VOLUME_SBML)
        assert model._varvol_conc_map, "expected a live-volume rescale map"
        rep = sbml_capability_report(model)
        assert any("time-varying" in note for note in rep["lossy"]), rep["lossy"]
        with pytest.raises(bngsim.ConversionError, match="time-varying"):
            write_sbml(model, tmp_path / "grow.xml", strict=True)
        # --allow-lossy still emits a best-effort (static-volume) document
        write_sbml(model, tmp_path / "grow.xml", strict=False)


# An assignment-rule compartment whose volume is time-INVARIANT (only constant
# parameters) vs time-VARYING (references the time csymbol). Same `tV := …`
# shape, opposite verdicts — the distinction that keeps constant-volume PBPK
# models (BIOMD1027–1039) convertible while refusing genuinely live volumes.
_AR_COMPARTMENT_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="ar">
    <listOfCompartments>
      <compartment id="cell" spatialDimensions="3" size="1" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="cell" initialAmount="10"
               hasOnlySubstanceUnits="true" boundaryCondition="true" constant="true"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="w" value="2" constant="true"/>
      <parameter id="frac" value="0.3" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="cell">
        <math xmlns="http://www.w3.org/1998/Math/MathML">{VOL}</math>
      </assignmentRule>
    </listOfRules>
  </model>
</sbml>
"""
_AR_CONST_VOL = "<apply><times/><ci>w</ci><ci>frac</ci></apply>"
_AR_TVAR_VOL = (
    "<apply><plus/><apply><times/><ci>w</ci><ci>frac</ci></apply>"
    '<csymbol encoding="text" '
    'definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol></apply>'
)


def test_constant_ar_compartment_volume_converts(tmp_path: Path) -> None:
    """An assignment-rule compartment whose volume is a constant expression
    (`w · frac`, both constant parameters) is NOT lossy — the report-time rescale
    is a no-op, so it round-trips as a plain static compartment. This is the
    PBPK organ-volume case (BIOMD1027–1039)."""
    sbml = _AR_COMPARTMENT_SBML.replace("{VOL}", _AR_CONST_VOL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        model = bngsim.Model.from_sbml_string(sbml)
        assert model._varvol_ar_conc_map, "expected an AR-compartment rescale map"
        assert not sbml_capability_report(model)["lossy"]
        write_sbml(model, tmp_path / "ar.xml", strict=True)  # must not raise


def test_time_varying_ar_compartment_volume_refused(tmp_path: Path) -> None:
    """The same compartment with a *time-varying* volume (references the time
    csymbol) IS refused under strict — its live volume cannot be a static SBML
    compartment. The function (not its constant shadow parameter) is the live
    definition, so the constancy check must resolve functions first."""
    sbml = _AR_COMPARTMENT_SBML.replace("{VOL}", _AR_TVAR_VOL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        model = bngsim.Model.from_sbml_string(sbml)
        rep = sbml_capability_report(model)
        assert any("varies in time" in note for note in rep["lossy"]), rep["lossy"]
        with pytest.raises(bngsim.ConversionError, match="varies in time"):
            write_sbml(model, tmp_path / "ar.xml", strict=True)


def test_species_dependent_ar_compartment_refused() -> None:
    """A benchmark model whose assignment-rule compartment volume depends on
    species (BIOMD856: `tV := mV + dV`, cell mass that grows over time) is
    refused — the contained species report `amount/tV(t)`, which a static volume
    cannot reproduce even though the kinetics round-trip."""
    src = _EVENTS / "BIOMD0000000856.xml"
    if not src.is_file():
        pytest.skip("BIOMD0000000856.xml not present")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        model = bngsim.Model.from_sbml(src)
        rep = sbml_capability_report(model)
        assert any("varies in time" in note for note in rep["lossy"]), rep["lossy"]


# ─── The faithful half: unit-volume SBML (incl amount-valued) round-trips ──


def test_amount_valued_unit_volume_roundtrips(tmp_path: Path) -> None:
    """BIOMD7 is mostly amount-valued (hasOnlySubstanceUnits) at unit volume.
    net2sbml reconstructs amount semantics, so write_sbml→from_sbml is exact."""
    src = _EVENTS / "BIOMD0000000007.xml"
    if not src.is_file():
        pytest.skip("BIOMD0000000007.xml not present")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        src_model = bngsim.Model.from_sbml(src)
        write_sbml(src_model, tmp_path / "b7.xml", strict=True)
        back = bngsim.Model.from_sbml(tmp_path / "b7.xml")
    assert _rhs_max_delta(src_model, back) <= 1e-9


# ─── L2 round-trip-identity seed (unblocks #217) ───────────────────────────


@pytest.mark.parametrize("name", ["simple_decay.net", "case_sensitivity.net", "mm_tqssa.net"])
def test_l2_net_sbml_net_identity(name: str, data_dir: Path, tmp_path: Path) -> None:
    """``.net`` → SBML → ``.net`` preserves structure and dynamics."""
    src = data_dir / name
    if not src.is_file():
        pytest.skip(f"tracked .net not present: {src}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        original = bngsim.Model.from_net(src)
        net_to_sbml(src, tmp_path / "rt.xml", validate=None, strict=True)
        sbml_to_net(tmp_path / "rt.xml", tmp_path / "rt.net", validate=None, strict=True)
        back = bngsim.Model.from_net(tmp_path / "rt.net")
    assert back.n_species == original.n_species
    # sbml_to_net re-encodes functional laws as per-species signed flux, so the raw
    # reaction count may grow; fold those fragments back to compare source topology.
    from bngsim.convert._validate import _effective_n_reactions

    assert _effective_n_reactions(back) == _effective_n_reactions(original)
    assert _rhs_max_delta(original, back) <= 1e-9


def test_l2_sbml_net_sbml_identity(data_dir: Path, tmp_path: Path) -> None:
    """SBML → ``.net`` → SBML preserves structure and dynamics (unit volume)."""
    src = data_dir / "BIOMD0000000003.xml"
    if not src.is_file():
        pytest.skip("BIOMD0000000003.xml not present")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        original = bngsim.Model.from_sbml(src)
        sbml_to_net(src, tmp_path / "rt.net", validate=None, strict=True)
        net_to_sbml(tmp_path / "rt.net", tmp_path / "rt.xml", validate=None, strict=True)
        back = bngsim.Model.from_sbml(tmp_path / "rt.xml")
    assert back.n_species == original.n_species
    assert back.n_reactions == original.n_reactions
    assert _rhs_max_delta(original, back) <= 1e-9


# ─── API + direct validation ───────────────────────────────────────────────


def test_write_sbml_in_memory_no_file(data_dir: Path) -> None:
    model = bngsim.Model.from_net(data_dir / "case_sensitivity.net")
    text = write_sbml(model, strict=True)
    assert "<sbml" in text and "<listOfReactions>" in text
    assert "<listOfSpecies>" in text


def test_validate_roundtrip_direct(data_dir: Path, tmp_path: Path) -> None:
    src = data_dir / "two_species_reversible.net"
    src_model = bngsim.Model.from_net(src)
    write_sbml(src_model, tmp_path / "m.xml", strict=True)
    sbml_model = bngsim.Model.from_sbml(tmp_path / "m.xml")
    rep = validate_roundtrip(src_model, sbml_model)
    assert rep.passed, rep.summary()


# ─── CLI ───────────────────────────────────────────────────────────────────


def test_cli_smoke(data_dir: Path, tmp_path: Path) -> None:
    from bngsim.convert._cli import net2sbml_main

    out = tmp_path / "out.xml"
    rc = net2sbml_main([str(data_dir / "case_sensitivity.net"), "-o", str(out), "--quiet"])
    assert rc == 0
    assert out.is_file()
    model = bngsim.Model.from_sbml(out)
    assert model.n_reactions == 2


def test_cli_tfun_strict_exits_nonzero(data_dir: Path, tmp_path: Path) -> None:
    src = data_dir / "tfun_time_indexed.net"
    if not src.is_file():
        pytest.skip("tfun_time_indexed.net not present")
    from bngsim.convert._cli import net2sbml_main

    rc = net2sbml_main([str(src), "-o", str(tmp_path / "t.xml")])
    assert rc == 1


# ─── Full L0–L4 gate wired into the converter (Option 1) ────────────────────


_SMITH = _BNGSIM / "benchmarks" / "models" / "sbml" / "Smith2013_BIOMD0000000474_petab.xml"


@pytest.mark.parametrize("name", _CLEAN_NETS)
def test_full_gate_passes_clean(name: str, data_dir: Path, tmp_path: Path) -> None:
    """``validate="full"`` runs the L0–L4 ladder and every hard gate (L0–L3)
    passes on the clean corpus; the verdict is attached and decides ``ok``."""
    from bngsim.convert import ConversionValidationReport

    src = data_dir / name
    if not src.is_file():
        pytest.skip(f"tracked .net not present: {src}")
    out = tmp_path / (src.stem + ".xml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = net_to_sbml(src, out, validate="full", strict=True)

    assert isinstance(report.validation, ConversionValidationReport)
    assert report.validation.direction == "net2sbml"
    assert report.ok, report.summary()
    for lv in ("L0", "L1", "L2", "L3"):
        level = report.validation.level(lv)
        assert level is not None and level.status == "pass", report.summary()


def test_full_gate_l4_is_non_gating(data_dir: Path, tmp_path: Path) -> None:
    """L4 may punt to *inconclusive* (MM tQSSA re-emitted in a non-smooth form)
    without blocking the conversion — only L0–L3 gate."""
    src = data_dir / "mm_tqssa.net"
    if not src.is_file():
        pytest.skip("mm_tqssa.net not present")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = net_to_sbml(src, tmp_path / "mm.xml", validate="full", strict=True)
    assert report.validation.level("L4").status == "inconclusive"
    assert report.validation.level("L4").gating is False
    assert report.ok  # gates pass; L4 never blocks


def test_full_gate_no_double_convert(data_dir: Path, tmp_path: Path) -> None:
    """The full gate grades the artifact the converter just produced — it must
    not re-run the forward conversion. ``from_net`` is the forward load, so it is
    called exactly once (the gate's L2/L3 use ``from_sbml`` / ``Simulator``)."""
    src = data_dir / "simple_decay.net"
    if not src.is_file():
        pytest.skip("simple_decay.net not present")
    orig = bngsim.Model.from_net
    calls = {"n": 0}

    def _counting_from_net(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        bngsim.Model.from_net = staticmethod(_counting_from_net)
        try:
            net_to_sbml(src, tmp_path / "d.xml", validate="full", strict=True)
        finally:
            bngsim.Model.from_net = staticmethod(orig)
    # 1 forward load + 1 reverse load (L2's SBML→.net→from_net round-trip) = 2,
    # never a second *forward* net→SBML conversion.
    assert calls["n"] == 2, calls["n"]


def test_unknown_validate_value_rejected(data_dir: Path) -> None:
    src = data_dir / "simple_decay.net"
    if not src.is_file():
        pytest.skip("simple_decay.net not present")
    with pytest.raises(ValueError, match="unknown validate"):
        net_to_sbml(src, validate="L9")


@pytest.mark.skipif(not _SMITH.is_file(), reason="Smith2013 benchmark not present")
def test_full_gate_catches_lossy_conversion(tmp_path: Path) -> None:
    """An unfaithful (lossy) conversion that L0/L1 accept is caught by the full
    gate's L2/L3. Smith2013 has amount-valued species in non-unit volumes the
    plain ``.net`` text cannot carry, so a best-effort sbml2net is structurally
    valid yet numerically wrong — and ``report.ok`` is False."""
    from bngsim.convert import sbml_to_net

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = sbml_to_net(_SMITH, tmp_path / "smith.net", validate="full", strict=False)
    assert report.validation is not None
    assert report.validation.level("L0").status == "pass"
    assert report.validation.level("L1").status == "pass"
    assert report.validation.level("L2").status == "fail"
    assert report.validation.level("L3").status == "fail"
    assert not report.ok, report.summary()


# ─── CLI: --gate flag ───────────────────────────────────────────────────────


def test_cli_gate_full_exit_zero(data_dir: Path, tmp_path: Path) -> None:
    from bngsim.convert._cli import net2sbml_main

    src = data_dir / "two_species_reversible.net"
    if not src.is_file():
        pytest.skip("two_species_reversible.net not present")
    out = tmp_path / "out.xml"
    rc = net2sbml_main([str(src), "-o", str(out), "--gate", "full", "--quiet"])
    assert rc == 0
    assert out.is_file()


def test_cli_gate_full_summary_lists_levels(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from bngsim.convert._cli import net2sbml_main

    src = data_dir / "simple_decay.net"
    if not src.is_file():
        pytest.skip("simple_decay.net not present")
    rc = net2sbml_main([str(src), "-o", str(tmp_path / "d.xml"), "--gate", "full"])
    out = capsys.readouterr().out
    assert rc == 0
    for lv in ("L0", "L1", "L2", "L3", "L4"):
        assert lv in out


@pytest.mark.skipif(not _SMITH.is_file(), reason="Smith2013 benchmark not present")
def test_cli_sbml2net_gate_full_lossy_exits_nonzero(tmp_path: Path) -> None:
    from bngsim.convert._cli import main as sbml2net_main

    rc = sbml2net_main(
        [
            str(_SMITH),
            "-o",
            str(tmp_path / "smith.net"),
            "--gate",
            "full",
            "--allow-lossy",
            "--quiet",
        ]
    )
    assert rc == 1


def test_cli_no_validate_alias_still_works(data_dir: Path, tmp_path: Path) -> None:
    """The deprecated ``--no-validate`` flag is a hidden alias for ``--gate none``."""
    from bngsim.convert._cli import net2sbml_main

    src = data_dir / "simple_decay.net"
    if not src.is_file():
        pytest.skip("simple_decay.net not present")
    rc = net2sbml_main([str(src), "-o", str(tmp_path / "d.xml"), "--no-validate", "--quiet"])
    assert rc == 0


# ─── Protocol-aware L3 horizon: source .bngl drives the gate (A) ────────────


_BNGL_T500 = (
    "begin model\nend model\ngenerate_network()\n"
    'simulate({method=>"ode",t_end=>500,n_steps=>50})\n'
)


def test_full_gate_uses_bngl_horizon(data_dir: Path, tmp_path: Path) -> None:
    """A source ``.bngl`` makes the full gate's L3 simulate over the model's own
    horizon (t_end=500), not the blanket grid; the protocol is attached."""
    src = data_dir / "simple_decay.net"
    if not src.is_file():
        pytest.skip("simple_decay.net not present")
    bngl = tmp_path / "decay.bngl"
    bngl.write_text(_BNGL_T500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = net_to_sbml(src, tmp_path / "d.xml", validate="full", strict=True, bngl=bngl)
    l3 = report.validation.level("L3")
    assert l3.metrics["t_span"] == [0.0, 500.0]
    assert l3.metrics["n_points"] == 51
    assert l3.metrics["horizon_source"] == "bngl protocol"
    assert report.protocol is not None
    assert report.protocol.primary_experiment().t_span == (0.0, 500.0)


def test_full_gate_falls_back_to_blanket_without_bngl(data_dir: Path, tmp_path: Path) -> None:
    src = data_dir / "simple_decay.net"
    if not src.is_file():
        pytest.skip("simple_decay.net not present")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = net_to_sbml(src, tmp_path / "d.xml", validate="full", strict=True)
    l3 = report.validation.level("L3")
    assert l3.metrics["t_span"] == [0.0, 100.0]  # blanket default
    assert "horizon_source" not in l3.metrics


def test_cli_bngl_sibling_auto_detected(data_dir: Path, tmp_path: Path) -> None:
    """When ``--bngl`` is omitted, a sibling ``<stem>.bngl`` is picked up."""
    from bngsim.convert._cli import net2sbml_main

    src = data_dir / "simple_decay.net"
    if not src.is_file():
        pytest.skip("simple_decay.net not present")
    net_copy = tmp_path / "decay.net"
    net_copy.write_text(src.read_text())
    (tmp_path / "decay.bngl").write_text(_BNGL_T500)
    out = tmp_path / "decay.xml"
    rc = net2sbml_main([str(net_copy), "-o", str(out), "--gate", "full", "--quiet"])
    assert rc == 0
    # The sibling horizon was used: reload and re-grade with the same bngl proves
    # the wiring, but here we just assert the conversion + gate succeeded.
    assert out.is_file()


def test_cli_bngl_missing_explicit_path_exits_two(data_dir: Path, tmp_path: Path) -> None:
    from bngsim.convert._cli import net2sbml_main

    src = data_dir / "simple_decay.net"
    if not src.is_file():
        pytest.skip("simple_decay.net not present")
    rc = net2sbml_main(
        [str(src), "-o", str(tmp_path / "d.xml"), "--bngl", str(tmp_path / "nope.bngl")]
    )
    assert rc == 2
