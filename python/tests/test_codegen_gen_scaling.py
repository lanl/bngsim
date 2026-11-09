"""GH #161 — codegen RHS *source generation* must stay ~linear in model size.

The serial ``.net`` generator (``generate_rhs_c`` / ``generate_sens_rhs_c``,
retired by #803) hid two accidentally-quadratic loops that made source
generation ~11 min on a 113k-reaction genome-scale model, dwarfing the
#160-sharded compile:

  1. ``set(param_idx.keys())`` was rebuilt once per reaction and handed to
     ``_classify_rate_law``, which never used it. Parameter count scales with
     reaction count, so that is O(n_reactions x n_params). (The classifier went
     with the ``.net`` parser in #803 step 4.)
  2. ``_translate_expr`` rebuilt its full identifier lookup (every param +
     observable + function, plus a ``_safe_c_name`` regex per observable and
     function) once per function body. On a model with ~18k functions and
     ~170k identifiers that is O(n_functions x n_identifiers).

The fixes drop the dead classifier argument and build the identifier lookup
once; both leave the emitted source byte-identical. The model-based generator
had the second quadratic too (``_translate_expr_to_c``), fixed the same way, and
since #803 it is the generator every ``.net`` model goes through.

These tests pin the properties, not a wall-clock number on one machine:
  * The translator does not take the input that invited its per-item rebuild —
    a deterministic root-cause guard.
  * An all-elementary model (exercises the RHS + twice-over sensitivity
    generator) and a function-heavy model (exercises function-body translation)
    each generate in about 10x the time of the same model a tenth the size, as a
    linear generator does, where a quadratic one takes about 100x.

The ratio is measured, not seconds (#1479). A 20 s budget took 1.7 s on an
M-series Mac and 22.7 s and 22.2 s on GitHub's ubuntu runners, though neither
change touched codegen. Two things make the ratio steadier than the seconds:
  * Both sizes run in the same process and are timed by its CPU time
    (``time.process_time``), not the wall clock. A slower CPU or a coverage
    tracer slows both sizes alike. pytest-xdist's other workers don't count at
    all. By the wall clock they did: with 32 busy processes on 16 CPUs the
    function-heavy ratio read 23.5-28.0, because a short run lands in a quiet
    moment more often than a long one, and by CPU time 11.4-12.3.
  * The cyclic collector is paused while timing. With it on, the elementary
    generator grew as about n^1.2 from 2.5k to 80k reactions (40k/4k: 16-20).
    With it paused it grew as n^1.00-1.04 (40k/4k: 10.3).
"""

from __future__ import annotations

import gc
import inspect
import math
import random
import re
import time

import bngsim
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
    reaction is functional. This drives function-body translation: each body
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


#: The largest a generator's time may grow when the model grows tenfold. Linear
#: is about 10 (10.3 and 10.5 measured, with the collector paused) and quadratic
#: about 100. The re-introduced GH #161 set-per-reaction loop read 80x and more.
MAX_GROWTH = 30.0


def _fastest_seconds(fn, repeats: int) -> float:
    """The fastest of ``repeats`` calls of ``fn``, in this process's CPU
    seconds, each with the cyclic collector paused. The fastest call, not the
    mean: cache contention on a shared runner only ever adds time."""
    best = math.inf
    for _ in range(repeats):
        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            t0 = time.process_time()
            fn()
            best = min(best, time.process_time() - t0)
        finally:
            if was_enabled:
                gc.enable()
    return best


def _growth(write, generate, tmp_path, n: int, *, small_repeats: int, large_repeats: int):
    """How many times longer ``generate(model)`` takes on the model ``write(path,
    n)`` makes than on the one ``write(path, n // 10)`` makes:
    ``(ratio, small_seconds, large_seconds, large_result)``."""
    times, result = [], None
    for size, repeats in ((n // 10, small_repeats), (n, large_repeats)):
        net = tmp_path / f"model_{size}.net"
        write(net, size)
        model = bngsim.Model.from_net(str(net))
        out = []
        times.append(_fastest_seconds(lambda m=model, o=out: o.append(generate(m)), repeats))
        result = out[-1]
    small, large = times
    return large / small, small, large, result


def test_translate_expr_to_c_takes_a_prebuilt_lookup():
    """Root-cause guard for the translator. ``_translate_expr_to_c`` must
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
    about 10x the time of a 4k-reaction one, not the ~100x of the pre-#161 O(n^2)
    generator, which needed about a minute at this size. Seconds are not bounded:
    the same generation took 1.7 s on an M-series Mac and 22.7 s on a GitHub
    runner (#1479)."""
    n_rxn = 40_000
    ratio, small, large, (source, has_sens) = _growth(
        lambda path, n: _elementary_net(path, n_species=n // 3, n_rxn=n),
        cg.generate_combined_from_model,
        tmp_path,
        n_rxn,
        small_repeats=5,
        large_repeats=2,
    )

    # Sanity: this is the path that ran both O(n^2) generators (RHS + sens).
    assert has_sens
    assert "bngsim_codegen_rhs" in source
    assert ratio < MAX_GROWTH, (
        f"generate_combined_from_model took {large:.2f}s for {n_rxn} reactions and "
        f"{small:.3f}s for {n_rxn // 10}, {ratio:.0f}x for 10x the model — source "
        "generation looks quadratic again (GH #161)."
    )


def test_function_heavy_generation_is_not_quadratic(tmp_path):
    """A model with 10k functions / observables / parameters generates RHS source
    in about 10x the time of a 1k one. Pre-#161 the translator rebuilt the full
    ~3n-identifier lookup (with a regex per observable/function) for each of the n
    function bodies, tens of seconds at 5000. The lookup is now built once, so
    the time grows with n. 10k and 1k, not 5000 and 500: 500 functions take
    about 6 ms, too near a timer's scatter to divide by."""
    n = 10_000
    ratio, small, large, source = _growth(
        _function_heavy_net,
        cg.generate_rhs_from_model,
        tmp_path,
        n,
        small_repeats=5,
        large_repeats=3,
    )

    # Every function body must have been emitted (the path that ran the O(n^2)).
    # One body per function — ``double func_<name> = …`` below the chunk gate, or
    # ``func[i] = …`` once the obs/func computation is sharded into NOINLINE blocks
    # (GH #165; 5000 reactions ⇒ chunked array form). Count both forms.
    n_bodies = source.count("double func_") + len(re.findall(r"\bfunc\[\d+\] =", source))
    assert n_bodies == n
    assert ratio < MAX_GROWTH, (
        f"generate_rhs_from_model took {large:.3f}s for {n} functions and "
        f"{small:.4f}s for {n // 10}, {ratio:.0f}x for 10x the model — "
        "function-body translation looks quadratic again (GH #161)."
    )
