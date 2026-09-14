"""Issue #532: a rate law carried by several reactions is derived once per model.

What a Functional rate law derives to depends on its text and on inputs that are
fixed for the whole model — the function map, the observable groups, the species
metadata, the constants, the writable volumes — and never on which reaction
carries it: each reaction's own stoichiometry is applied afterwards, by the C++
scatter at attach and by the C scatter in the compiled Jacobian. Sharing is
common. A ``.net`` function drives every reaction its rule generated, and the SBML
loader emits a reaction with non-integer stoichiometry as one reaction per
species, each carrying the whole kinetic law.

The five Smallbone 2013 linlog BioModels (BIOMD0000000469–473) have a biomass
reaction with 65 reactants and 3 products. Its 5,413-character law was derived
once for each of the 68 per-species reactions it becomes: 92% of a 29 s
derivation that ran into the 20 s budget, so whether the model kept its analytical
Jacobian depended on machine load. ``attach_functional_jacobian`` and the compiled
Jacobian's reconstruction (``_codegen._functional_jacobian_groups``) now derive
each distinct rate law once per path and hand the terms to every reaction that
carries it.

The caches live for one derivation of one model, so nothing in them can outlive
its inputs. What would make them wrong is a builder that starts reading something
reaction-specific; the signature test below fails first.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import _codegen
from bngsim import _jacobian as J

# fShared drives two reactions with a reactant (the per-observable path) and two
# without (the per-species path); fOther drives one more.
_NET_SHARED = """\
begin parameters
    1 k 0.7
    2 Km 3.0
end parameters
begin species
    1 A() 2.0
    2 B() 1.5
    3 C() 0.5
    4 D() 0.25
end species
begin functions
    1 fShared() k*A_tot*B_tot*exp(-C_tot)/(Km+A_tot)
    2 fOther() k*C_tot/(1+D_tot)
end functions
begin reactions
    1 1 3 fShared #A_to_C
    2 2 4 fShared #B_to_D
    3 0 2 fShared #source_B
    4 0 1 fShared #source_A
    5 3 1 fOther #C_to_A
end reactions
begin groups
    1 A_tot 1
    2 B_tot 2
    3 C_tot 3
    4 D_tot 4
end groups
"""


def _species(sid: str, conc: str) -> str:
    return (
        f'<species id="{sid}" compartment="cell" initialConcentration="{conc}" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
    )


def _ref(sid: str, stoich: str) -> str:
    return f'<speciesReference species="{sid}" stoichiometry="{stoich}" constant="true"/>'


_MATH = '<math xmlns="http://www.w3.org/1998/Math/MathML">{}</math>'
_V_OVER_CELL = _MATH.format("<apply><divide/><ci>v</ci><ci>cell</ci></apply>")

# growth (0.5 A + 0.25 B -> 1.5 C) has non-integer stoichiometry, so the loader
# emits it as three per-species reactions carrying one law; two more reactions
# have laws of their own.
_SBML_SPLIT = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="split_growth">
    <listOfCompartments><compartment id="cell" size="2" constant="true"/></listOfCompartments>
    <listOfSpecies>
      {_species("A", "2")}{_species("B", "1.5")}{_species("C", "0.5")}{_species("D", "0.25")}
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.7" constant="true"/>
      <parameter id="Km" value="3" constant="true"/>
      <parameter id="v" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="v">{
    _MATH.format(
        "<apply><divide/><apply><times/><ci>k</ci><ci>A</ci><ci>B</ci>"
        "<apply><exp/><apply><minus/><ci>C</ci></apply></apply></apply>"
        "<apply><plus/><ci>Km</ci><ci>A</ci></apply></apply>"
    )
}</assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="growth" reversible="false">
        <listOfReactants>{_ref("A", "0.5")}{_ref("B", "0.25")}</listOfReactants>
        <listOfProducts>{_ref("C", "1.5")}</listOfProducts>
        <kineticLaw>{_V_OVER_CELL}</kineticLaw>
      </reaction>
      <reaction id="r2" reversible="false">
        <listOfReactants>{_ref("B", "1")}</listOfReactants>
        <listOfProducts>{_ref("D", "1")}</listOfProducts>
        <kineticLaw>{_V_OVER_CELL}</kineticLaw>
      </reaction>
      <reaction id="r3" reversible="false">
        <listOfReactants>{_ref("C", "1")}</listOfReactants>
        <listOfProducts>{_ref("A", "1")}</listOfProducts>
        <kineticLaw>{_V_OVER_CELL}</kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "BNGSIM_JAC_DERIV_BUDGET_S",
        "BNGSIM_ANALYTICAL_FUNCTIONAL_JAC",
        "BNGSIM_JAC_NO_SELFCHECK",
        "BNGSIM_JAC_SELFCHECK_DENSE_MAX",
        "BNGSIM_JAC_SELFCHECK_SAMPLE",
        "BNGSIM_NO_CODEGEN_JAC",
    ):
        monkeypatch.delenv(var, raising=False)


def _load(kind: str, tmp_path: Path, text: str | None = None) -> bngsim.Model:
    if kind == "sbml":
        return bngsim.Model.from_sbml_string(text or _SBML_SPLIT)
    path = tmp_path / "shared.net"
    path.write_text(text or _NET_SHARED)
    return bngsim.Model.from_net(str(path))


def _laws_by_path(m: bngsim.Model) -> dict[bool, list[str]]:
    """Each path's distinct rate laws; True is the per-observable path."""
    laws: dict[bool, list[str]] = {True: [], False: []}
    for rxn in m._core.functional_jacobian_context()["functional_reactions"]:
        path = bool(rxn["apply_species_factor"]) and len(rxn["reactant_idx0"]) > 0
        if rxn["rate_expr"] not in laws[path]:
            laws[path].append(rxn["rate_expr"])
    return laws


def _recording(monkeypatch: pytest.MonkeyPatch, name: str) -> dict[str, list]:
    """Wrap ``bngsim._jacobian.<name>`` so every call is recorded as rate law ->
    [results]. Both callers look the builder up on the module at call time."""
    seen: dict[str, list] = {}
    real = getattr(J, name)

    def wrapper(rate_expr, *args, **kwargs):
        out = real(rate_expr, *args, **kwargs)
        seen.setdefault(rate_expr, []).append(out)
        return out

    monkeypatch.setattr(J, name, wrapper)
    return seen


class _RecordingCore:
    """The model's core, with the terms handed to ``set_functional_jacobian`` kept."""

    def __init__(self, core) -> None:
        self._core = core
        self.terms = None

    def __getattr__(self, name):
        return getattr(self._core, name)

    def set_functional_jacobian(self, terms):
        self.terms = terms
        return self._core.set_functional_jacobian(terms)


def _central(m: bngsim.Model, y: np.ndarray, h: float = 1e-6) -> np.ndarray:
    n = y.size
    D = np.zeros((n, n))
    for j in range(n):
        s = h * max(abs(y[j]), 1.0)
        yp, ym = y.copy(), y.copy()
        yp[j] += s
        ym[j] -= s
        D[:, j] = (np.asarray(m.rhs(yp)) - np.asarray(m.rhs(ym))) / (2.0 * s)
    return D


# ─── The attach ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["net", "sbml"])
def test_each_distinct_rate_law_is_derived_once_per_path(kind, tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    m = _load(kind, tmp_path)
    laws = _laws_by_path(m)
    n_reactions = len(m._core.functional_jacobian_context()["functional_reactions"])
    per_observable = _recording(monkeypatch, "build_per_observable_terms")
    per_species = _recording(monkeypatch, "build_per_species_terms")

    assert m.prepare_analytical_jacobian() is True, m.analytical_jacobian_status

    assert {law: len(calls) for law, calls in per_observable.items()} == dict.fromkeys(
        laws[True], 1
    )
    assert {law: len(calls) for law, calls in per_species.items()} == dict.fromkeys(laws[False], 1)
    assert len(laws[True]) + len(laws[False]) < n_reactions, "the fixture shares no rate law"


@pytest.mark.parametrize("kind", ["net", "sbml"])
def test_every_reaction_receives_the_terms_it_would_derive_alone(kind, tmp_path, monkeypatch):
    """Exactness: the terms handed to C++ for each reaction are the terms that
    reaction's rate law derives to when derived on its own, uncached."""
    _clear_env(monkeypatch)
    m = _load(kind, tmp_path)
    core = _RecordingCore(m._core)
    assert J.attach_functional_jacobian(core) is True

    ctx = m._core.functional_jacobian_context()
    func_map = dict(ctx["function_map"])
    obs_groups = {n: [(int(s), float(f)) for s, f in g] for n, g in ctx["observables"]}
    obs_idx = {n: i for i, (n, _g) in enumerate(ctx["observables"])}
    smeta = {i: (bool(a), float(v)) for i, (a, v) in enumerate(ctx["species_meta"])}
    volumes = {i: n for i, n in enumerate(ctx.get("species_volume_param") or ()) if n}
    consts = set(ctx["constant_names"])
    expected = []
    for rxn in ctx["functional_reactions"]:
        law = rxn["rate_expr"]
        if rxn["apply_species_factor"] and rxn["reactant_idx0"]:
            obs_terms = J.build_per_observable_terms(law, func_map, set(obs_groups), consts)
            expected.append((rxn["rxn_idx"], True, [(obs_idx[n], e) for n, e in obs_terms]))
        else:
            sp_terms = J.build_per_species_terms(
                law, func_map, obs_groups, smeta, consts, None, volumes
            )
            expected.append((rxn["rxn_idx"], False, [(int(j), e) for j, e in sp_terms]))
    assert core.terms == expected


@pytest.mark.parametrize("kind", ["net", "sbml"])
def test_the_attached_jacobian_matches_finite_differences(kind, tmp_path, monkeypatch):
    """Each reaction still scatters into its own rows with its own stoichiometry."""
    _clear_env(monkeypatch)
    m = _load(kind, tmp_path)
    assert m.prepare_analytical_jacobian() is True
    rng = np.random.default_rng(532)
    for _ in range(4):
        y = rng.uniform(0.2, 2.5, len(m.species_names))
        Jy = m.jacobian(y)
        assert Jy.source == "analytical"
        np.testing.assert_allclose(np.asarray(Jy), _central(m, y), rtol=1e-6, atol=1e-9)


def _net_many(n: int, shared: bool) -> str:
    """``n`` sources of ``n`` species: one law for all of them, or one law each."""
    species = "\n".join(f"    {i} X{i}() 1.0" for i in range(1, n + 1))
    groups = "\n".join(f"    {i} X{i}_tot {i}" for i in range(1, n + 1))
    if shared:
        functions = "    1 fAll() k*X1_tot*X2_tot/(1+X1_tot)"
        reactions = "\n".join(f"    {i} 0 {i} fAll" for i in range(1, n + 1))
    else:
        functions = "\n".join(
            f"    {i} f{i}() k*X1_tot*X2_tot/({i}+X1_tot)" for i in range(1, n + 1)
        )
        reactions = "\n".join(f"    {i} 0 {i} f{i}" for i in range(1, n + 1))
    return (
        "begin parameters\n    1 k 0.1\nend parameters\n"
        f"begin species\n{species}\nend species\n"
        f"begin functions\n{functions}\nend functions\n"
        f"begin reactions\n{reactions}\nend reactions\n"
        f"begin groups\n{groups}\nend groups\n"
    )


def test_reactions_sharing_a_rate_law_spend_the_budget_once(tmp_path, monkeypatch):
    """Twelve reactions whose derivation takes 0.25 s each against a 1 s budget:
    sharing one rate law they attach, and with one law each they run out of time —
    which is what the five linlog models did before."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("BNGSIM_JAC_DERIV_BUDGET_S", "1.0")
    real = J.build_per_species_terms

    def slow(*args, **kwargs):
        time.sleep(0.25)
        return real(*args, **kwargs)

    monkeypatch.setattr(J, "build_per_species_terms", slow)

    shared = _load("net", tmp_path, _net_many(12, shared=True))
    assert shared.prepare_analytical_jacobian() is True, shared.analytical_jacobian_status

    distinct = _load("net", tmp_path, _net_many(12, shared=False))
    assert distinct.prepare_analytical_jacobian() is False
    assert "exceeded its 1.0s budget" in distinct.analytical_jacobian_status


def test_a_shared_rate_law_that_declines_still_says_why(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    text = _NET_SHARED.replace("k*A_tot*B_tot*exp(-C_tot)/(Km+A_tot)", "k*sign(A_tot)").replace(
        "fOther() k*C_tot/(1+D_tot)", "fOther() k*sign(A_tot)"
    )
    m = _load("net", tmp_path, text)
    assert m.prepare_analytical_jacobian() is False
    assert m.analytical_jacobian_status.startswith(
        "declined: rate law 'k*sign(A_tot)' applies sign"
    )


# ─── The compiled Jacobian's reconstruction ─────────────────────────────────


@pytest.mark.parametrize("kind", ["net", "sbml"])
def test_the_compiled_jacobian_derives_each_rate_law_once_and_places_it_everywhere(
    kind, tmp_path, monkeypatch
):
    """Once per distinct law and path, and every reaction's C group carries its
    law's derivative: the same lines a reaction with that law to itself would get."""
    _clear_env(monkeypatch)
    m = _load(kind, tmp_path)
    assert m.prepare_analytical_jacobian() is True
    laws = _laws_by_path(m)
    per_species = _recording(monkeypatch, "build_per_species_c")
    per_observable = _recording(monkeypatch, "differentiate_rate_law_c")

    src = _codegen.generate_jacobian_from_model(m)

    assert src is not None
    assert {law: len(calls) for law, calls in per_species.items()} == dict.fromkeys(laws[False], 1)
    assert {law: len(calls) for law, calls in per_observable.items()} == dict.fromkeys(
        laws[True], 1
    )
    for rxn in m._core.functional_jacobian_context()["functional_reactions"]:
        law, idx = rxn["rate_expr"], int(rxn["rxn_idx"])
        if rxn["apply_species_factor"] and rxn["reactant_idx0"]:
            (derivs,) = per_observable[law]
            block = src.split(f"/* per-observable rxn {idx} func ", 1)[1].split("/* per-", 1)[0]
            for k, (_obs, c) in enumerate(derivs):
                assert f"double d{k} = {c};" in block, (idx, k)
        else:
            (terms,) = per_species[law]
            for col, c in terms:
                block = src.split(f"/* per-species rxn {idx} col {col} */", 1)[1]
                assert block.split("\n", 2)[1].strip() == f"double dj = {c};", (idx, col)


# ─── The invariant the caches rest on ────────────────────────────────────────


@pytest.mark.parametrize(
    "builder, parameters",
    [
        (
            "build_per_species_terms",
            ["rate_expr", "func_map", "obs_groups", "species_amount", "constant_names"]
            + ["deadline", "species_volume_sym"],
        ),
        (
            "build_per_observable_terms",
            ["rate_expr", "func_map", "observable_names", "constant_names", "deadline"],
        ),
        (
            "build_per_species_c",
            ["rate_expr", "func_map", "obs_groups", "species_amount", "constant_names"]
            + ["resolve_symbol", "deadline", "species_volume_sym"],
        ),
        (
            "differentiate_rate_law_c",
            ["rate_expr", "func_map", "observable_names", "constant_names", "resolve_symbol"]
            + ["deadline"],
        ),
    ],
)
def test_only_the_rate_law_reaches_a_builder_from_the_reaction(builder, parameters):
    """The caches key on the rate law and the path because every other argument
    of these builders is fixed for the whole model. A new parameter has to be one
    of those too, or become part of the key in attach_functional_jacobian and
    _codegen._functional_jacobian_groups — this fails so that the decision gets made."""
    assert list(inspect.signature(getattr(J, builder)).parameters) == parameters


# ─── The corpus model of the issue ───────────────────────────────────────────

_B469 = Path("parity_checks/rr_parity/models/BIOMD0000000469/BIOMD0000000469_url.xml")


@pytest.mark.skipif(
    not _B469.exists(), reason="rr_parity corpus model BIOMD0000000469 not present"
)
def test_biomd469_derives_its_biomass_law_once_and_keeps_its_jacobian(monkeypatch):
    _clear_env(monkeypatch)
    m = bngsim.Model.from_sbml(str(_B469))
    rxns = m._core.functional_jacobian_context()["functional_reactions"]
    per_species = _recording(monkeypatch, "build_per_species_terms")

    assert m.prepare_analytical_jacobian() is True, m.analytical_jacobian_status

    assert sum(r["rate_expr"] == "r_2584/cell" for r in rxns) == 68
    assert len(per_species["r_2584/cell"]) == 1
    assert len(per_species) == len({r["rate_expr"] for r in rxns})
    assert m.analytical_jacobian_status == "complete"
