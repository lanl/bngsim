"""BNG2.pl round-trip validation of cBNGL output (GH #224).

cBNGL faithfulness is validated by an **independent oracle**: write the ``.bngl``,
run ``BNG2.pl generate_network`` to flatten it to a ``.net``, reload that through
:meth:`bngsim.Model.from_net`, and compare the reloaded ODE right-hand side to the
source's. There is no in-tree cBNGL reader by design — a Python re-implementation of
BNG's volume bake would no longer be an independent check. This module is shared by
the production gate (``sbml_to_bngl(validate="bng2")``), the test suite, and the
corpus sweep so all three measure faithfulness the same way.

The RHS is probed at several **t > 0** times (not just t=0): BNG2.pl rewrites
``>=``/``<=`` against numeric literals to ``>``/``<``, so a time-pulse that is "on"
at exactly t=0 in the source reads "off" in BNG — a measure-zero boundary that is
trajectory-faithful but trips a t=0-only probe. Sampling generic t>0 instants both
dodges that artifact and actually exercises time-dependent forcing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from bngsim._exceptions import ConversionError

if TYPE_CHECKING:
    from bngsim._model import Model

# Generic (non-round) instants spanning ~5 decades. Non-round so they never land on a
# pulse edge (BNG's >=→> rewrite makes edges differ only at the exact boundary, a
# measure-zero set generic floats avoid); spread wide so time-dependent forcing at
# any scale is exercised. RHS of an autonomous model is time-independent, so these
# reduce to a plain state probe for it.
_PROBE_TIMES = (0.0137, 0.37, 4.1, 53.0, 6700.0)


def find_bng2() -> Path | None:
    """Locate ``BNG2.pl`` via ``$BNGPATH`` (the parity-suite convention) or ``PATH``.

    Returns ``None`` if not found. Deliberately does **not** hardcode an absolute
    path — the production gate's contract is "set ``$BNGPATH`` or put ``BNG2.pl`` on
    ``PATH``"; the test suite supplies its own local fallback.
    """
    bngpath = os.environ.get("BNGPATH")
    cands: list[Path] = []
    if bngpath:
        cands += [Path(bngpath) / "BNG2.pl", Path(bngpath)]
    which = shutil.which("BNG2.pl")
    if which:
        cands.append(Path(which))
    for c in cands:
        if c.is_file():
            return c
    return None


def _strip_pattern(name: str) -> str:
    """``@comp::Mol()`` (from_net) → ``Mol`` (the molecule = source species name)."""
    s = name.split("::")[-1] if "::" in name else name
    return s[:-2] if s.endswith("()") else s


def _name_aligned_perm(src: Model, reloaded: Model) -> list[int]:
    """Map each source species index to the reloaded species index by molecule name.

    BNG2.pl may reorder/rename species, so alignment is by the writer's sanitized
    molecule name rather than position. Raises :class:`ConversionError` if a source
    species has no counterpart in the reloaded network.
    """
    from bngsim.convert._bngl_writer import _molecule_names

    sd = src._core.codegen_data()
    rl_names = [_strip_pattern(s["name"]) for s in reloaded._core.codegen_data()["species"]]
    _, src_mol = _molecule_names(
        sd["species"],
        sd["parameters"],
        sd["functions"],
        {f["name"] for f in sd["functions"]} | {o["name"] for o in sd["observables"]},
    )
    pos = {n: i for i, n in enumerate(rl_names)}
    missing = [n for n in src_mol if n not in pos]
    if missing:
        raise ConversionError(
            f"cannot align {len(missing)} species after BNG2.pl round-trip (e.g. {missing[:5]})"
        )
    return [pos[n] for n in src_mol]


def _rhs_delta(src: Model, reloaded: Model, perm: list[int], times=_PROBE_TIMES) -> float:
    """Largest scale-relative ``|Δ dy/dt|`` over the shared initial state and a few
    aligned probe states, evaluated at each of ``times``. Non-finite cells
    (out-of-domain probes) are skipped."""
    import numpy as np

    y0 = np.asarray(src.get_state(), dtype=float)
    rng = np.random.default_rng(0)
    states = [y0] + [np.abs(y0 * (0.3 + rng.random(len(y0)) * 2.0) + 1e-9) for _ in range(5)]
    worst = 0.0
    for t in times:
        for y in states:
            a = np.asarray(src._core._eval_rhs(t, y.tolist()), dtype=np.float64)
            b = np.asarray(reloaded._core._eval_rhs(t, y[perm].tolist()), dtype=np.float64)
            bb = np.empty_like(b)
            bb[perm] = b
            finite = np.isfinite(a) & np.isfinite(bb)
            if not finite.any():
                continue
            scale = max(float(np.abs(a[finite]).max(initial=0.0)), 1.0)
            worst = max(worst, float(np.abs(a[finite] - bb[finite]).max() / scale))
    return worst


def roundtrip_rhs_delta(
    model: Model,
    bngl_text: str,
    *,
    stem: str = "model",
    timeout: int = 300,
    bng2: Path | None = None,
) -> tuple[float, int]:
    """Run ``bngl_text`` through ``BNG2.pl generate_network`` → ``.net`` →
    :meth:`Model.from_net`, and return ``(max_rhs_delta, reloaded_n_species)``.

    ``max_rhs_delta`` is the scale-relative RHS difference vs ``model`` (see
    :func:`_rhs_delta`). Raises :class:`ConversionError` when ``BNG2.pl`` is
    unavailable, times out, produces no network, or the reloaded network's species
    cannot be name-aligned — i.e. whenever faithfulness cannot be established.
    """
    from bngsim._model import Model

    bng2 = bng2 or find_bng2()
    if bng2 is None:
        raise ConversionError(
            "validate='bng2' needs BNG2.pl — set $BNGPATH or put BNG2.pl on PATH"
        )

    d = Path(tempfile.mkdtemp())
    try:
        bp = d / f"{stem}.bngl"
        bp.write_text(bngl_text + "\ngenerate_network({overwrite=>1})\n")
        try:
            proc = subprocess.run(
                [str(bng2), "--outdir", str(d), str(bp)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise ConversionError(
                f"BNG2.pl generate_network timed out after {timeout}s — cannot "
                "validate (network may be too combinatorial to flatten)"
            ) from e
        net = d / f"{stem}.net"
        if not net.is_file():
            tail = ((proc.stdout or "")[-400:] + (proc.stderr or "")[-200:]).strip()
            raise ConversionError(
                f"BNG2.pl produced no network — the generated .bngl did not build:\n{tail}"
            )
        reloaded = Model.from_net(net)
        perm = _name_aligned_perm(model, reloaded)
        return _rhs_delta(model, reloaded, perm), reloaded.n_species
    finally:
        shutil.rmtree(d, ignore_errors=True)
