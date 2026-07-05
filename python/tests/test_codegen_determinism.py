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
