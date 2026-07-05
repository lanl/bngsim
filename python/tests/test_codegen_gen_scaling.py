"""GH #161 — codegen RHS *source generation* must stay ~linear in model size.

The serial generator (``generate_rhs_c`` / ``generate_sens_rhs_c``) hid two
accidentally-quadratic loops that made source generation ~11 min on a
113k-reaction genome-scale model, dwarfing the #160-sharded compile:

  1. ``set(param_idx.keys())`` was rebuilt once per reaction and handed to
     ``_classify_rate_law``, which never used it. Parameter count scales with
     reaction count, so that is O(n_reactions x n_params).
  2. ``_translate_expr`` rebuilt its full identifier lookup (every param +
     observable + function, plus a ``_safe_c_name`` regex per observable and
     function) once per function body. On a model with ~18k functions and
     ~170k identifiers that is O(n_functions x n_identifiers).

The fixes drop the dead classifier argument and build the identifier lookup
once via ``_build_ident_lookup``; both leave the emitted source byte-identical.

These tests pin the properties, not a wall-clock number on one machine:
  * Neither hot function takes the input that invited its per-item rebuild —
    deterministic root-cause guards.
  * An all-elementary model (exercises the RHS + twice-over sensitivity
    generator) and a function-heavy model (exercises ``_translate_expr``) each
    generate with large headroom under a budget the quadratic could not have
    met. Pre-fix needed ~minutes; the budgets are tens of seconds, so only a
    re-quadratic-ied generator trips them — machine-speed variance does not.
"""

from __future__ import annotations

import inspect
import random
import re
import time

from bngsim import _codegen as cg


def _elementary_net(path, n_species: int, n_rxn: int, seed: int = 1234) -> None:
    """Write an all-elementary mass-action .net so sensitivity RHS is emitted
    too (its all-elementary scan + rxn_data build were the two extra O(n^2)
    sites, on top of the RHS one)."""
    rng = random.Random(seed)
    lines = ["begin parameters"]
    for i in range(n_rxn):
        lines.append(f"{i + 1} k{i} {round(abs(rng.gauss(0.5, 0.3)) + 0.02, 5)}")
    lines += ["end parameters", "begin species"]
    for i in range(n_species):
        lines.append(f"{i + 1} S{i}() {round(abs(rng.gauss(2.0, 1.0)) + 0.1, 5)}")
    lines += ["end species", "begin reactions"]
    for i in range(n_rxn):
        a, b = rng.randrange(n_species) + 1, rng.randrange(n_species) + 1
        c = rng.randrange(n_species) + 1
        lines.append(f"{i + 1} {a},{b} {c} k{i} #_R{i + 1}")
    lines += ["end reactions", "begin groups", "end groups"]
    path.write_text("\n".join(lines) + "\n")


def _function_heavy_net(path, n: int, seed: int = 7) -> None:
    """Write a model with ~n parameters, observables, and functions, where every
    reaction is functional. This drives ``_translate_expr``: each function body
    is translated against a lookup of all ~3n identifiers, so a per-call rebuild
    is O(n_functions x n_identifiers) (GH #161 quadratic #2)."""
    rng = random.Random(seed)
    n_sp = max(2, n // 3)
    lines = ["begin parameters"]
    for i in range(n):
        lines.append(f"{i + 1} k{i} {round(abs(rng.gauss(0.5, 0.3)) + 0.02, 5)}")
    lines += ["end parameters", "begin species"]
    for i in range(n_sp):
        lines.append(f"{i + 1} S{i}() {round(abs(rng.gauss(2.0, 1.0)) + 0.1, 5)}")
    lines += ["end species", "begin functions"]
    # Each body references a parameter and an observable, so the lookup matters.
    for i in range(n):
        lines.append(f"{i + 1} f{i}() k{i}*O{i % n} + 2.0")
    lines += ["end functions", "begin reactions"]
    for i in range(n):
        a = rng.randrange(n_sp) + 1
        c = rng.randrange(n_sp) + 1
        lines.append(f"{i + 1} {a} {c} f{i} #_R{i + 1}")  # functional rate law
    lines += ["end reactions", "begin groups"]
    for i in range(n):  # n observables, each a single-species group
        lines.append(f"{i + 1} O{i} {(i % n_sp) + 1}")
    lines += ["end groups"]
    path.write_text("\n".join(lines) + "\n")


def test_classify_rate_law_does_not_take_param_name_set():
    """Root-cause guard: the classifier consults only func_names. Re-adding a
    parameter-name argument invites the per-reaction ``set(param_idx)`` rebuild
    that made generation O(n^2) (GH #161)."""
    params = list(inspect.signature(cg._classify_rate_law).parameters)
    assert params == ["rate_law", "func_names"], (
        f"_classify_rate_law signature is {params!r}; it must not accept the "
        "parameter-name set (GH #161 — building it per reaction is quadratic)."
    )


def test_translate_expr_takes_a_prebuilt_lookup():
    """Root-cause guard: ``_translate_expr`` rewrites against a prebuilt lookup,
    not the raw index maps. Passing the maps invites the per-call lookup rebuild
    (all params + observables + functions, with a regex per name) that made
    function-body translation O(n^2) (GH #161)."""
    params = list(inspect.signature(cg._translate_expr).parameters)
    assert params == ["expr", "lookup"], (
        f"_translate_expr signature is {params!r}; it must take a prebuilt "
        "lookup (GH #161 — rebuilding it per function body is quadratic)."
    )


def test_translate_expr_to_c_takes_a_prebuilt_lookup():
    """Same guard for the model-based translator. ``_translate_expr_to_c`` must
    take a prebuilt lookup, not the per-name maps — rebuilding the combined
    (~245k-entry, species included) table per body was the model-based GH #161
    quadratic that left ``generate_rhs_from_model`` unfinished after >5 min on
    the genome-scale model."""
    params = list(inspect.signature(cg._translate_expr_to_c).parameters)
    assert params == ["expr", "lookup"], (
        f"_translate_expr_to_c signature is {params!r}; it must take a prebuilt "
        "lookup (GH #161 — rebuilding it per body is quadratic)."
    )


def test_large_model_generation_is_not_quadratic(tmp_path):
    """A 40k-reaction all-elementary model generates RHS + sensitivity source in
    well under the budget. The pre-#161 O(n^2) generator needed ~a minute at this
    size; the budget is 20 s with >20x headroom on a normal machine, so it trips
    only on a genuine re-quadratic regression, not on CI jitter."""
    net = tmp_path / "genome.net"
    n_rxn = 40_000
    _elementary_net(net, n_species=n_rxn // 3, n_rxn=n_rxn)

    t0 = time.perf_counter()
    source, has_sens = cg.generate_combined_c(str(net))
    elapsed = time.perf_counter() - t0

    # Sanity: this is the path that ran both O(n^2) generators (RHS + sens).
    assert has_sens
    assert "bngsim_codegen_rhs" in source
    assert elapsed < 20.0, (
        f"generate_combined_c took {elapsed:.1f}s for {n_rxn} reactions — "
        "source generation looks quadratic again (GH #161)."
    )


def test_function_heavy_generation_is_not_quadratic(tmp_path):
    """A model with 5000 functions / observables / parameters generates RHS
    source well under the budget. Pre-#161 ``_translate_expr`` rebuilt the full
    ~15k-identifier lookup (with a regex per observable/function) for each of the
    5000 function bodies — tens of seconds. The lookup is now built once, so this
    is sub-second; the 15 s budget has >15x headroom and trips only on a genuine
    regression, not CI jitter."""
    net = tmp_path / "funcs.net"
    n = 5000
    _function_heavy_net(net, n)

    t0 = time.perf_counter()
    source = cg.generate_rhs_c(str(net))
    elapsed = time.perf_counter() - t0

    # Every function body must have been emitted (the path that ran the O(n^2)).
    # One body per function — ``double func_<name> = …`` below the chunk gate, or
    # ``func[i] = …`` once the obs/func computation is sharded into NOINLINE blocks
    # (GH #165; 5000 reactions ⇒ chunked array form). Count both forms.
    n_bodies = source.count("double func_") + len(re.findall(r"\bfunc\[\d+\] =", source))
    assert n_bodies == n
    assert elapsed < 15.0, (
        f"generate_rhs_c took {elapsed:.1f}s for {n} functions — function-body "
        "translation looks quadratic again (GH #161 _translate_expr)."
    )
