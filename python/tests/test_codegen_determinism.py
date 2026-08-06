"""GH #111 follow-up — codegen C output must be deterministic across processes.

The model-based codegen hashes its generated C source to key the compiled-`.so`
cache. A mass-action reaction's *product* index list was built by iterating a
``set`` union of species IDs (``_classify_mass_action_ast``), whose order is
``PYTHONHASHSEED``-randomized — so the emitted ``ydot[p] += rate`` order, the C
text, and its SHA-256 differed every process. The cache therefore never hit
(every load recompiled the multi-MB RHS) and builds were not reproducible.

Sorting the product indices fixes it without any numerical change (the order of
independent ``ydot[p] += rate`` accumulations is irrelevant; products do not enter
the rate expression). This test pins determinism by generating the codegen hash
in several child processes with different ``PYTHONHASHSEED`` values and asserting
they all agree — it fails if any set/dict-ordered iteration leaks into the C.

GH #65 adds the other axis of determinism: the Elementary ``bngsim_dfdp``
emission is pinned to its exact text, so a change to the sensitivity emitter that
is supposed to be Elementary-neutral cannot quietly shift it. ``bngsim_dfdp``
gained the ability to take ``obs[]``/``func[]`` for the Functional ``∂f/∂p`` of
#66; an Elementary derivative ``∂(k·sf·∏y^m)/∂k`` is written purely in
``p[]``/``y[]``, so it must keep the original five-parameter signature and the
original body.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

# A mass-action reaction with several products so the product set-union has many
# orderings; pre-fix this reliably permuted under different hash seeds. Built
# programmatically so the (long) SBML attribute lines live in string content, not
# in over-length source lines.
_PRODUCTS = ["B", "C", "D", "E", "F", "G"]
_SPECIES = ["A", *_PRODUCTS]


def _species_xml(sid: str, conc: str) -> str:
    return (
        f'<species id="{sid}" compartment="c" initialConcentration="{conc}" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
    )


def _product_ref(sid: str) -> str:
    return f'<speciesReference species="{sid}" stoichiometry="1" constant="true"/>'


_SBML = "".join(
    [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core"',
        ' level="3" version="2"><model>',
        '<listOfCompartments><compartment id="c" size="1" constant="true"/>',
        "</listOfCompartments><listOfSpecies>",
        *[_species_xml(s, "10" if s == "A" else "0") for s in _SPECIES],
        "</listOfSpecies>",
        '<listOfParameters><parameter id="k" value="0.5" constant="true"/>',
        "</listOfParameters>",
        '<listOfReactions><reaction id="R1" reversible="false">',
        "<listOfReactants>",
        _product_ref("A"),  # one reactant A, same element shape
        "</listOfReactants><listOfProducts>",
        *[_product_ref(s) for s in _PRODUCTS],
        "</listOfProducts><kineticLaw>",
        '<math xmlns="http://www.w3.org/1998/Math/MathML">',
        "<apply><times/><ci>k</ci><ci>A</ci></apply></math>",
        "</kineticLaw></reaction></listOfReactions></model></sbml>",
    ]
)

_CHILD = textwrap.dedent(
    """
    import os, sys, hashlib
    os.environ["BNGSIM_NO_CODEGEN"] = "1"
    import bngsim
    from bngsim import _codegen
    m = bngsim.Model.from_sbml_string(sys.stdin.read())
    c_source, _ = _codegen.generate_combined_from_model(m)
    sys.stdout.write(hashlib.sha256(c_source.encode()).hexdigest())
    """
)


def _codegen_hash_with_seed(seed: int) -> str:
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        input=_SBML,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, f"child failed (seed={seed}):\n{proc.stderr}"
    out = proc.stdout.strip()
    assert len(out) == 64, f"unexpected child output (seed={seed}): {proc.stdout!r}"
    return out


def test_codegen_hash_is_pythonhashseed_independent():
    hashes = {seed: _codegen_hash_with_seed(seed) for seed in (0, 1, 2, 3)}
    distinct = set(hashes.values())
    assert len(distinct) == 1, f"codegen hash varies with PYTHONHASHSEED: {hashes}"


# ─── GH #68: the same leak, in the per-observable derivative ordering ─────────
#
# ``differentiate_rate_law`` iterated ``observable_names`` — a *set* — so a rate
# law reading two or more observables ordered its ``{observable: ∂rate/∂obs}``
# result, and with it every ``double d0``/``d1`` temporary and the ``v[j]``
# column each is scattered into, by ``PYTHONHASHSEED``. Same class of bug as the
# product-index one above, one layer down, and the saturable twin
# (``_saturable_jacobian.differentiate_rate_law_native``) had already sorted for
# it. The single-observable laws that dominate the corpus hid it: with one
# derivative there is only one ordering.
#
# It reached the *sensitivity* RHS when GH #68 admitted condition-bearing
# Functional laws, which is where it was caught — the four biggest models it
# unblocked re-hashed on every process and so could never hit their own ``.so``.

_TWO_OBSERVABLE_NET = """\
begin parameters
    1 ka     0.7  # Constant
    2 kb     1.3  # Constant
    3 kc     0.9  # Constant
    4 c0     2.0  # Constant
end parameters
begin functions
    1 drive() c0*sin(ka*Atot)*Btot + kc*Ctot + kb*Atot*Ctot
end functions
begin species
    1 A() 3.0
    2 B() 5.0
    3 C() 2.0
    4 D() 0.0
end species
begin reactions
    1 1 4 drive #_R1
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
    3 Ctot                 3
end groups
"""

_FUNCTIONAL_CHILD = textwrap.dedent(
    """
    import hashlib, sys, tempfile, os
    import bngsim
    from bngsim import _codegen
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.net")
    open(p, "w").write(sys.stdin.read())
    m = bngsim.Model.from_net(p)
    src = _codegen.generate_sens_from_model(m, functional=True)
    assert src is not None, "the fixture must reach the Functional sens emitter"
    sys.stdout.write(hashlib.sha256(src.encode()).hexdigest())
    """
)


def test_functional_sens_rhs_hash_is_pythonhashseed_independent():
    """A three-observable Functional rate law, through the sensitivity emitter.
    Both halves are covered at once: the ∂f/∂p switch and the fused ``J·v``,
    which is where the per-observable ordering actually lands.

    ``sin()`` is load-bearing: it puts the law outside the saturable family, so
    ``differentiate_rate_law_c`` takes the **sympy** branch. The native branch
    already sorted, so a saturable fixture passes this test with the bug still
    in place."""
    hashes = {}
    for seed in (0, 1, 2, 3):
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        proc = subprocess.run(
            [sys.executable, "-c", _FUNCTIONAL_CHILD],
            input=_TWO_OBSERVABLE_NET,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert proc.returncode == 0, f"child failed (seed={seed}):\n{proc.stderr}"
        hashes[seed] = proc.stdout.strip()
        assert len(hashes[seed]) == 64, f"unexpected output (seed={seed}): {proc.stdout!r}"
    assert len(set(hashes.values())) == 1, (
        f"Functional sens RHS hash varies with PYTHONHASHSEED: {hashes}"
    )


# ─── GH #96: the same leak again, in the #198 output-sensitivity partials ─────
#
# Third instance of one bug. ``differentiate_expression_output_partials``
# iterated ``free`` — the *set* of a function body's free symbols — so the order
# its ``{symbol: ∂f/∂symbol}`` maps were built in, and with it the order the
# chain-rule terms are emitted into ``bngsim_codegen_output_sens``, tracked
# ``PYTHONHASHSEED``. Measured on **121 of the 585** ``.net`` corpus models,
# whose #198 evaluator therefore hashed differently every process and could never
# hit its content-addressed ``.so``.
#
# It surfaced while A/B-ing an unrelated change: 121 models' emitted C "changed",
# and the same two hashes came back under different seeds with identical code.
# Any before/after comparison of this emitter is meaningless until it is pinned,
# which is the other reason it belongs in the suite rather than in a note.
#
# A one-symbol function body cannot show it — with one partial there is only one
# ordering — so the fixture below gives every function several referenced
# symbols of several kinds (species, observable, parameter, earlier function).

_MULTI_SYMBOL_NET = """\
begin parameters
    1 ka     0.7  # Constant
    2 kb     1.3  # Constant
    3 kc     0.9  # Constant
    4 kd     2.1  # Constant
end parameters
begin functions
    1 base()  ka*Atot + kb*Btot + kc*Ctot
    2 mixed() kd*base() + ka*Atot*Btot + kb*Ctot
end functions
begin species
    1 A() 3.0
    2 B() 5.0
    3 C() 2.0
    4 D() 0.0
end species
begin reactions
    1 1 4 mixed #_R1
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
    3 Ctot                 3
end groups
"""

_OUTPUT_SENS_CHILD = textwrap.dedent(
    """
    import hashlib, sys, tempfile, os
    import bngsim
    from bngsim import _codegen
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.net")
    open(p, "w").write(sys.stdin.read())
    m = bngsim.Model.from_net(p)
    src = _codegen.generate_output_sens_from_model(m)
    assert src is not None, "the fixture must reach the #198 output-sens emitter"
    sys.stdout.write(hashlib.sha256(src.encode()).hexdigest())
    """
)


def test_output_sens_hash_is_pythonhashseed_independent():
    hashes = {}
    for seed in (0, 1, 2, 3):
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        proc = subprocess.run(
            [sys.executable, "-c", _OUTPUT_SENS_CHILD],
            input=_MULTI_SYMBOL_NET,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert proc.returncode == 0, f"child failed (seed={seed}):\n{proc.stderr}"
        hashes[seed] = proc.stdout.strip()
        assert len(hashes[seed]) == 64, f"unexpected output (seed={seed}): {proc.stdout!r}"
    assert len(set(hashes.values())) == 1, (
        f"#198 output-sens hash varies with PYTHONHASHSEED: {hashes}"
    )


# ─── GH #65: the Elementary sensitivity emission is frozen ────────────────────

# The full emitted ``bngsim_dfdp`` for the model above: one mass-action reaction
# A → B+C+D+E+F+G with rate constant p[1], so ∂f/∂k is `v = y[0]` scattered by
# the net stoichiometry. Written out rather than hashed so a diff on this test
# shows *what* moved.
# (#170) The switch gained three cases. The reaction's rate constant used to be
# the bare parameter with the compartment volume folded numerically into the
# Elementary scalar; it is now the derived ``_rateLaw_<rid> = kf * kr / (C/V_0)``,
# so ``∂f/∂C`` exists (case 0) and the two constants reach the rate through
# issue #43's chain rule (cases 1 and 2) instead of directly. Same RHS, same
# numbers at the nominal point — the ratio is exactly 1.0 there — and one more
# column that used to be structurally absent.
_EXPECTED_ELEMENTARY_DFDP = """\
static void bngsim_dfdp(int iP, double t, const double* y,
                        const double* p, double* dfdp_out) {
    memset(dfdp_out, 0, N_SPECIES * sizeof(double));

    double v;
    switch (iP) {
    case 0:
        v = (-p[2]*p[1]/pow(p[0], 2)) * y[0];
        dfdp_out[0] -= v;
        dfdp_out[1] += v;
        dfdp_out[2] += v;
        dfdp_out[3] += v;
        dfdp_out[4] += v;
        dfdp_out[5] += v;
        dfdp_out[6] += v;
        break;
    case 1:
        v = (p[2]/p[0]) * y[0];
        dfdp_out[0] -= v;
        dfdp_out[1] += v;
        dfdp_out[2] += v;
        dfdp_out[3] += v;
        dfdp_out[4] += v;
        dfdp_out[5] += v;
        dfdp_out[6] += v;
        break;
    case 2:
        v = (p[1]/p[0]) * y[0];
        dfdp_out[0] -= v;
        dfdp_out[1] += v;
        dfdp_out[2] += v;
        dfdp_out[3] += v;
        dfdp_out[4] += v;
        dfdp_out[5] += v;
        dfdp_out[6] += v;
        break;
    case 3:
        v = y[0];
        dfdp_out[0] -= v;
        dfdp_out[1] += v;
        dfdp_out[2] += v;
        dfdp_out[3] += v;
        dfdp_out[4] += v;
        dfdp_out[5] += v;
        dfdp_out[6] += v;
        break;
    default:
        break;  /* parameter not a rate constant - dfdp = 0 */
    }

}
"""


def _emitted_dfdp() -> str:
    import bngsim
    from bngsim import _codegen

    model = bngsim.Model.from_sbml_string(_SBML)
    c_source, has_sens = _codegen.generate_combined_from_model(model)
    assert has_sens, "an all-Elementary model must still get an analytical sens RHS"
    start = c_source.index("static void bngsim_dfdp")
    return c_source[start : c_source.index("\n}\n", start) + 3]


def test_elementary_dfdp_emission_is_unchanged():
    """The #65 guard. The signature line is the load-bearing part: gaining
    ``const double* obs`` / ``const double* func`` here would mean an Elementary
    model started paying for context it has no derivative to use."""
    assert _emitted_dfdp() == _EXPECTED_ELEMENTARY_DFDP


def test_elementary_sens_rhs_carries_no_observable_context():
    """The driver half: no obs[]/func[] arrays are declared or filled, and the
    dispatch keeps its five arguments."""
    import bngsim
    from bngsim import _codegen

    model = bngsim.Model.from_sbml_string(_SBML)
    sens = _codegen.generate_sens_from_model(model)
    assert sens is not None
    assert "    bngsim_dfdp(iP, t, y, p, dfdp);" in sens
    assert "double obs[" not in sens
    assert "double func[" not in sens
    assert "sens_obs_blk" not in sens and "sens_func_blk" not in sens
