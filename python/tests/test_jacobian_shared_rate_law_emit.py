"""Issue #1551: on the per-species path, a rate law shared by several reactions
is printed once in the compiled Jacobian, and its derivative is scattered into
every reaction that carries it.

#539 (issue #532) made ``_functional_jacobian_groups`` derive each distinct rate
law once per path. The emitter still printed the result once per reaction. The
five Smallbone 2013 linlog BioModels (BIOMD0000000469–473) have a biomass reaction
the SBML loader splits into 68 per-species reactions sharing one law with 65
column derivatives of about 4 KB of C each. Printed for each of the 68 carriers,
they were 18.1 MB of BIOMD0000000469's 19.6 MB source, which took 13–22 minutes
to compile at ``-O0``, so its parity job timed out against every reference.

What is pinned here:

* **Printed once.** A per-species law emits one ``double dj`` per column, however
  many reactions carry it. These fail on the code before the fix, which printed
  them per carrier. A law with one carrier keeps the group it always had.
* **Placed everywhere.** The compiled Jacobian, dense and sparse, flat and
  chunked, matches the interpreted assembly (``_dense_analytical_jacobian``, the
  FD-self-checked oracle) and a finite difference, so a term scattered into the
  wrong row, or not at all, fails. In the ``.net`` fixture a source of another
  law sits between two carriers and writes the entry (A, C) that the second one
  also writes, so that entry sums in a different order than the interpreted
  assembly does. With two terms the order cannot change the rounding, so no
  fixture here observes a reordering effect; the tolerance is what would absorb
  one.
* **The sensitivity RHS's ``J·v``,** the other consumer of the same groups,
  still equals the compiled ``J`` times ``v``.

The per-observable (``.net``) path still emits one group per reaction; see the
comment in ``_functional_jacobian_groups``.
"""

from __future__ import annotations

import ctypes
import hashlib
import re
import shutil
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg

pytest.importorskip("sympy")
_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")

# fShared is carried by two per-species reactions, the sources 6 and 8, with the
# source 7 of fOther between them writing the entry (A, C) that source 8 writes
# too. The per-observable reactions (1, 3, 5, 9 for fShared, 2 and 4 for fOther)
# go through the path that still emits a group per reaction, and they write the
# same rows, so a misplaced per-species term cannot hide behind them. Reaction 3
# has a reactant of multiplicity 2 and reaction 9 reads the fixed species $S.
_NET = """\
begin parameters
    1 k 0.7
    2 Km 3.0
    3 k2 0.4
end parameters
begin species
    1 A() 2.0
    2 B() 1.5
    3 C() 0.5
    4 D() 0.25
    5 $S() 1.2
end species
begin functions
    1 fShared() k*A_tot*B_tot*exp(-C_tot)/(Km+A_tot)
    2 fOther() k2*C_tot/(1+D_tot)
end functions
begin reactions
    1 1 3 fShared #A_to_C
    2 1 2 fOther #A_to_B
    3 1,1 4 fShared #2A_to_D
    4 3 1 fOther #C_to_A
    5 2,3 4 fShared #B_C_to_D
    6 0 2 fShared #source_B
    7 0 1 fOther #source_A_other
    8 0 1 fShared #source_A
    9 5 3 fShared #S_to_C
end reactions
begin groups
    1 A_tot 1
    2 B_tot 2
    3 C_tot 3
    4 D_tot 4
end groups
"""


def _species(sid: str, conc: float) -> str:
    return (
        f'<species id="{sid}" compartment="cell" initialConcentration="{conc}" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
    )


def _ref(sid: str, stoich: str) -> str:
    return f'<speciesReference species="{sid}" stoichiometry="{stoich}" constant="true"/>'


def _sbml_biomass(n: int) -> str:
    """A biomass-style reaction ``0.5 S1 + … + 0.5 Sn -> 1.5 X`` with a law over
    every reactant, which the SBML loader emits as n + 1 per-species reactions
    sharing one law with n column derivatives, the shape of BIOMD0000000469's
    r_2584. A saturable reaction S1 -> X with a law of its own writes the entries
    (S1, *) and (X, *) the carriers also write."""
    sids = [f"S{i}" for i in range(1, n + 1)]
    factors = "".join(
        f"<apply><divide/><ci>{s}</ci><apply><plus/><ci>K</ci><ci>{s}</ci></apply></apply>"
        for s in sids
    )
    law = f"<apply><times/><ci>k</ci>{factors}</apply>"
    math = '<math xmlns="http://www.w3.org/1998/Math/MathML">{}</math>'
    species = "".join(_species(s, 1.0 + 0.3 * i) for i, s in enumerate(sids)) + _species("X", 0.4)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="biomass_{n}">
    <listOfCompartments><compartment id="cell" size="2" constant="true"/></listOfCompartments>
    <listOfSpecies>{species}</listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.9" constant="true"/>
      <parameter id="K" value="1.7" constant="true"/>
      <parameter id="k2" value="0.3" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="biomass" reversible="false">
        <listOfReactants>{"".join(_ref(s, "0.5") for s in sids)}</listOfReactants>
        <listOfProducts>{_ref("X", "1.5")}</listOfProducts>
        <kineticLaw>{math.format(f"<apply><times/><ci>cell</ci>{law}</apply>")}</kineticLaw>
      </reaction>
      <reaction id="convert" reversible="false">
        <listOfReactants>{_ref("S1", "1")}</listOfReactants>
        <listOfProducts>{_ref("X", "1")}</listOfProducts>
        <kineticLaw>{
        math.format(
            "<apply><divide/><apply><times/><ci>cell</ci><ci>k2</ci><ci>S1</ci><ci>X</ci></apply>"
            "<apply><plus/><ci>K</ci><ci>X</ci></apply></apply>"
        )
    }</kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def _net_chain(n: int) -> str:
    """A sparse chain A1 -> … -> An whose every step is driven by one function
    over A1 (the per-observable path), with sources of A2..A6 driven by it too
    (the per-species path) and a step of another law between each pair of
    carriers. ``n >= 50`` with a few entries per column routes to KLU."""
    lines = ["begin parameters", "    1 k 0.6", "    2 Km 2.0", "    3 kd 0.05"]
    lines += ["end parameters", "begin species"]
    lines += [f"    {i} A{i}() {1.0 + 0.05 * i}" for i in range(1, n + 1)]
    lines += ["end species", "begin functions", "    1 fShared() k*Atot/(Km+Atot)"]
    lines += ["    2 fOther() kd*Btot", "end functions", "begin reactions"]
    r = 1
    for i in range(1, n):
        lines.append(f"    {r} {i} {i + 1} fShared")
        r += 1
        lines.append(f"    {r} {i + 1} {i} fOther")
        r += 1
    for i in range(2, 7):
        lines.append(f"    {r} 0 {i} fShared")
        r += 1
        lines.append(f"    {r} 0 {i} fOther")
        r += 1
    lines += ["end reactions", "begin groups", "    1 Atot 1", "    2 Btot 2", "end groups", ""]
    return "\n".join(lines)


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "BNGSIM_JAC_DERIV_BUDGET_S",
        "BNGSIM_ANALYTICAL_FUNCTIONAL_JAC",
        "BNGSIM_JAC_NO_SELFCHECK",
        "BNGSIM_NO_CODEGEN_JAC",
        "BNGSIM_CODEGEN_CHUNK",
        "BNGSIM_CODEGEN_CHUNK_SIZE",
    ):
        monkeypatch.delenv(var, raising=False)


def _net_model(tmp_path: Path, text: str, name: str = "shared.net") -> bngsim.Model:
    path = tmp_path / name
    path.write_text(text)
    m = bngsim.Model.from_net(str(path))
    assert m.prepare_analytical_jacobian() is True, m.analytical_jacobian_status
    return m


def _sbml_model(text: str) -> bngsim.Model:
    m = bngsim.Model.from_sbml_string(text)
    assert m.prepare_analytical_jacobian() is True, m.analytical_jacobian_status
    return m


def _carriers(m: bngsim.Model) -> dict[tuple[bool, str], list[int]]:
    """(per-observable path?, rate law) -> the reactions carrying it."""
    out: dict[tuple[bool, str], list[int]] = {}
    for rxn in m._core.functional_jacobian_context()["functional_reactions"]:
        path = bool(rxn["apply_species_factor"]) and len(rxn["reactant_idx0"]) > 0
        out.setdefault((path, rxn["rate_expr"]), []).append(int(rxn["rxn_idx"]))
    return out


# ─── Printed once ────────────────────────────────────────────────────────────


def _law_with(carriers: dict, n: int) -> str:
    (law,) = [law for (path, law), rxns in carriers.items() if not path and len(rxns) == n]
    return law


def test_a_per_species_law_prints_each_column_derivative_once():
    """n + 1 carriers of one law over n reactants: n derivatives, not n·(n + 1)."""
    n = 12
    m = _sbml_model(_sbml_biomass(n))
    carriers = _carriers(m)
    law = _law_with(carriers, n + 1)
    src = cg.generate_jacobian_from_model(m)
    assert src is not None
    # The biomass law's n columns, plus the two columns (S1, X) of `convert`.
    assert src.count("double dj = ") == n + 2
    first = carriers[(False, law)][0]
    note = f", and the {n} other reactions carrying its rate law */"
    assert src.count(f"/* per-species rxn {first} col ") == n
    assert src.count(note) == n
    for idx in carriers[(False, law)][1:]:
        assert f"/* per-species rxn {idx} col " not in src


def test_a_law_with_one_carrier_keeps_the_group_it_always_had():
    m = _sbml_model(_sbml_biomass(3))
    (convert,) = [rxns for (path, law), rxns in _carriers(m).items() if len(rxns) == 1]
    src = cg.generate_jacobian_from_model(m)
    assert src is not None
    groups = re.findall(
        rf"/\* per-species rxn {convert[0]} col \d+ \*/\n        double dj = ", src
    )
    assert len(groups) == 2


def test_the_net_fixture_merges_its_per_species_sources_only(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    m = _net_model(tmp_path, _NET)
    carriers = _carriers(m)
    shared = "k*A_tot*B_tot*exp(-C_tot)/(Km+A_tot)"
    assert len(carriers[(False, shared)]) == 2
    assert len(carriers[(True, shared)]) == 4
    src = cg.generate_jacobian_from_model(m)
    assert src is not None
    # fShared's three columns once for its two sources, fOther's two for its one.
    assert src.count("double dj = ") == 3 + 2
    # Every per-observable reaction still has a group of its own.
    assert src.count("/* per-observable rxn ") == 6
    assert src.count("double d0 = ") == 6


# ─── Placed everywhere ───────────────────────────────────────────────────────


class _UserData(ctypes.Structure):
    # CodegenUserData: param_values, tfun_ctx, tfun_eval. No tables here.
    _fields_ = [
        ("param_values", ctypes.POINTER(ctypes.c_double)),
        ("tfun_ctx", ctypes.c_void_p),
        ("tfun_eval", ctypes.c_void_p),
    ]


def _call_jac(core, so, symbol: str, y: np.ndarray, size: int) -> np.ndarray:
    lib = ctypes.CDLL(str(so))
    fn = getattr(lib, symbol)
    dp = ctypes.POINTER(ctypes.c_double)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_double, dp, dp, ctypes.c_void_p]
    params = [p["value"] for p in core.codegen_data()["parameters"]]
    pbuf = (ctypes.c_double * len(params))(*params)
    ud = _UserData(param_values=ctypes.cast(pbuf, dp), tfun_ctx=None, tfun_eval=None)
    out = (ctypes.c_double * size)()
    assert fn(0.0, (ctypes.c_double * len(y))(*y), out, ctypes.byref(ud)) == 0
    return np.array(out, dtype=float)


def _compiled(m: bngsim.Model, src: str | None = None):
    """The compiled Jacobian as a function of the state returning J[row][col],
    whichever symbol the model got. ``src`` is the model's Jacobian source, if
    the caller already generated it."""
    core = m._core
    so = cg.prepare_model_codegen(m)
    assert so is not None
    ns = core.n_species
    plan = core.codegen_jacobian_plan()
    if src is None:
        src = cg.generate_jacobian_from_model(m) or ""
    if "bngsim_codegen_jac_sparse" in src:
        cp, ri = plan["col_ptrs"], plan["row_indices"]

        def sparse(y):
            vals = _call_jac(core, so, "bngsim_codegen_jac_sparse", y, int(plan["nnz"]))
            dense = np.zeros((ns, ns))
            for col in range(ns):
                for k in range(int(cp[col]), int(cp[col + 1])):
                    dense[int(ri[k]), col] = vals[k]
            return dense

        return sparse
    assert "int bngsim_codegen_jac(" in src
    return lambda y: _call_jac(core, so, "bngsim_codegen_jac", y, ns * ns).reshape(ns, ns).T


def _interpreted(m: bngsim.Model, y: np.ndarray) -> np.ndarray:
    ns = m._core.n_species
    return np.array(m._core._dense_analytical_jacobian(0.0, list(y))).reshape(ns, ns).T


def _fd(m: bngsim.Model, y: np.ndarray) -> np.ndarray:
    core = m._core
    f0 = np.array(core._eval_rhs(0.0, list(y)), dtype=float)
    out = np.zeros((y.size, y.size))
    for j in range(y.size):
        h = 1e-6 * max(1.0, abs(y[j]))
        yp = y.copy()
        yp[j] += h
        out[:, j] = (np.array(core._eval_rhs(0.0, list(yp)), dtype=float) - f0) / h
    return out


def _assert_compiled_matches(m: bngsim.Model, seed: int) -> None:
    compiled = _compiled(m)
    rng = np.random.default_rng(seed)
    for _ in range(3):
        y = rng.uniform(0.2, 4.0, m._core.n_species)
        comp = compiled(y)
        interp = _interpreted(m, y)
        assert np.count_nonzero(interp) > 0
        scale = np.maximum(np.maximum(np.abs(comp), np.abs(interp)), 1e-9)
        assert float((np.abs(comp - interp) / scale).max()) < 1e-12
        fd = _fd(m, y)
        scale = np.maximum(np.maximum(np.abs(comp), np.abs(fd)), 1e-3)
        assert float((np.abs(comp - fd) / scale).max()) < 1e-4


@needs_cc
@pytest.mark.parametrize("chunk", ["off", "on"])
def test_the_net_fixture_matches_the_interpreted_jacobian(tmp_path, monkeypatch, chunk):
    _clear_env(monkeypatch)
    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", chunk)
    m = _net_model(tmp_path, _NET)
    src = cg.generate_jacobian_from_model(m)
    assert src is not None and "carrying its rate law" in src
    assert ("NOINLINE" in src) == (chunk == "on")
    _assert_compiled_matches(m, 1551)


@needs_cc
@pytest.mark.parametrize("chunk", ["off", "on"])
def test_the_biomass_fixture_matches_the_interpreted_jacobian(monkeypatch, chunk):
    _clear_env(monkeypatch)
    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", chunk)
    m = _sbml_model(_sbml_biomass(8))
    src = cg.generate_jacobian_from_model(m)
    assert src is not None and "carrying its rate law" in src
    _assert_compiled_matches(m, 8)


@needs_cc
@pytest.mark.parametrize("chunk", ["off", "on"])
def test_a_sparse_chain_of_carriers_matches_the_interpreted_jacobian(tmp_path, monkeypatch, chunk):
    _clear_env(monkeypatch)
    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", chunk)
    m = _net_model(tmp_path, _net_chain(60), "chain.net")
    plan = m._core.codegen_jacobian_plan()
    if not plan["has_klu"]:
        pytest.skip("KLU sparse solver not built into this core")
    src = cg.generate_jacobian_from_model(m)
    assert src is not None and "bngsim_codegen_jac_sparse" in src
    # The five sources of each law: one column each, printed once.
    assert src.count("double dj = ") == 2
    _assert_compiled_matches(m, 60)


# ─── The other consumer: the sensitivity RHS's J·v ──────────────────────────


class _SensUserData(ctypes.Structure):
    _fields_ = [
        ("param_values", ctypes.POINTER(ctypes.c_double)),
        ("plist", ctypes.POINTER(ctypes.c_int)),
        ("n_sens", ctypes.c_int),
    ]


@needs_cc
@pytest.mark.parametrize("chunk", ["off", "on"])
def test_jac_vec_is_the_compiled_jacobian_times_v(tmp_path, monkeypatch, chunk):
    """``bngsim_jac_vec`` is built from the same merged groups with the matvec
    fused into the scatter; ``ySdot(v) - ySdot(0)`` isolates it."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", chunk)
    m = _net_model(tmp_path, _NET)
    src, has_sens = cg.generate_combined_from_model(m)
    assert has_sens and "carrying its rate law" in src
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)
    so = cg.compile_rhs(src, hashlib.sha256(src.encode()).hexdigest()[:16])
    lib = ctypes.CDLL(str(so))
    dp = ctypes.POINTER(ctypes.c_double)
    sens = lib.bngsim_codegen_sens_rhs
    sens.restype = ctypes.c_int
    sens.argtypes = [ctypes.c_int, ctypes.c_double, dp, dp, ctypes.c_int, dp, dp]
    sens.argtypes += [ctypes.c_void_p, dp, dp]
    core = m._core
    ns = core.n_species
    params = [p["value"] for p in core.codegen_data()["parameters"]]
    pbuf = (ctypes.c_double * len(params))(*params)

    def sens_rhs(y, ys):
        ud = _SensUserData(param_values=pbuf, plist=(ctypes.c_int * 1)(0), n_sens=1)
        out = (ctypes.c_double * ns)()
        tmp = [(ctypes.c_double * ns)() for _ in range(3)]
        args = (ctypes.c_double * ns)(*y), tmp[0], 0, (ctypes.c_double * ns)(*ys), out
        assert sens(1, 0.0, *args, ctypes.byref(ud), tmp[1], tmp[2]) == 0
        return np.array(out)

    rng = np.random.default_rng(67)
    for _ in range(2):
        y = rng.uniform(0.2, 4.0, ns)
        J = _call_jac(core, so, "bngsim_codegen_jac", y, ns * ns).reshape(ns, ns).T
        np.testing.assert_allclose(J, _interpreted(m, y), rtol=1e-12, atol=1e-12)
        for v in (*np.eye(ns), rng.normal(size=ns)):
            jv = sens_rhs(y, v) - sens_rhs(y, np.zeros(ns))
            np.testing.assert_allclose(jv, J @ v, rtol=1e-12, atol=1e-12)


# ─── The corpus model of the issue ───────────────────────────────────────────

_B469 = Path("parity_checks/rr_parity/models/BIOMD0000000469/BIOMD0000000469_url.xml")


@needs_cc
@pytest.mark.skipif(
    not _B469.exists(), reason="rr_parity corpus model BIOMD0000000469 not present"
)
def test_biomd469_prints_its_biomass_law_once_and_compiles_a_correct_jacobian(monkeypatch):
    """19.2 MB of Jacobian before, 18.1 MB of it the biomass law's 65 column
    derivatives once per carrier; under 1 MB after, and still right."""
    _clear_env(monkeypatch)
    m = bngsim.Model.from_sbml(str(_B469))
    assert m.prepare_analytical_jacobian() is True, m.analytical_jacobian_status
    carriers = _carriers(m)[(False, "r_2584/cell")]
    assert len(carriers) == 68

    src = cg.generate_jacobian_from_model(m)

    assert src is not None
    assert len(src) < 1_000_000
    first = re.findall(r"/\* per-species rxn (\d+) col \d+, and the 67 other reactions", src)
    assert len(first) == 65 and set(first) == {str(carriers[0])}
    for idx in carriers[1:]:
        assert f"/* per-species rxn {idx} col " not in src
    y = np.asarray(m._core.get_initial_state(), dtype=float)
    comp = _compiled(m, src)(y)
    interp = _interpreted(m, y)
    assert np.count_nonzero(interp) > 0
    scale = np.maximum(np.maximum(np.abs(comp), np.abs(interp)), 1e-9)
    assert float((np.abs(comp - interp) / scale).max()) < 1e-12
