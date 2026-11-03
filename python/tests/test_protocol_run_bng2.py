"""Issue #1366: ``run_protocol`` reproduces BNG2.pl's own output for whole protocols.

Each case replays a corpus file's actions on the network ``benchmarks`` ships for
it (the seed state the file generates) and compares with what BNG2.pl 2.9.3
wrote running the file whole, kept in ``tests/data/bng2_protocol`` (see its README).
So the comparison needs no BioNetGen here, and runs on every leg.

The controls are the closed-form tests in ``test_protocol_run.py``; these hold
the replay to the reference engine on real protocols: an equilibration then a
dose (``egfr_path``), a rest, a dose and a washout continued from it (``nfkb``),
and nine scans with parameter sets and resets between them (``wofsy_goldstein``).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim.convert import parse_bngl_protocol

_REPO = Path(__file__).resolve().parents[2]
# Beside the suite's other data under tests/data, which run_tests.sh's stand-in
# links; python/tests holds only the modules it copies.
_DATA = _REPO / "tests" / "data" / "bng2_protocol"
# Both engines run CVODE at run_network's 1e-8 tolerances; a difference below
# 100x the absolute one is inside both.
_RTOL = 1e-5
_FLOOR = 1e-6


def _replay(name: str) -> bngsim.ProtocolResult:
    model = bngsim.Model.from_net(str(_REPO / "benchmarks/models/net/ode" / f"{name}.net"))
    with warnings.catch_warnings():
        # The parse flags what it cannot carry for SED-ML; the replay checks itself.
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        spec = parse_bngl_protocol(
            _REPO / "benchmarks/models/bngl/ode" / f"{name}.bngl", strict=False
        )
    return bngsim.run_protocol(model, spec)


def _read(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    rows = [[float(x) for x in line.split()] for line in lines[1:] if line.strip()]
    return lines[0].lstrip("#").split(), np.array(rows)


def _column(result, name: str) -> np.ndarray:
    """An observable, or a function BNG2.pl printed beside them (print_functions)."""
    if name in result.observable_names:
        return np.asarray(result.observables[name])
    return np.asarray(result.expressions[name])


def _close(ours, ref) -> None:
    ours = np.asarray(ours, dtype=float)
    off = np.abs(ours - ref) > _FLOOR
    np.testing.assert_allclose(ours[off], ref[off], rtol=_RTOL)


@pytest.mark.parametrize("name", ["egfr_path", "nfkb"])
def test_a_phased_protocol_ends_as_bng2_ends_it(name):
    """BNG2.pl's ``<name>.gdat`` holds the file's last run that does not
    continue, and the runs that continue it, appended."""
    out = _replay(name)
    runs = [r for r in out if r.kind == "simulate" and not r.label]
    start = max(i for i, r in enumerate(runs) if not r.experiment.extra.get("continue"))
    results = [r.result for r in runs[start:]]
    header, ref = _read(_DATA / f"{name}.gdat")
    time = np.concatenate([r.time for r in results])
    # A continued run repeats its first time, which BNG2.pl writes once.
    keep = np.r_[True, np.diff(time) != 0]
    assert keep.sum() == len(ref)
    _close(time[keep], ref[:, 0])
    for j, column in enumerate(header[1:], start=1):
        values = np.concatenate([_column(r, column) for r in results])
        _close(values[keep], ref[:, j])


def test_every_scan_of_a_scan_protocol_is_bng2s():
    """Each point's row is the observables at the end of its run; between the
    scans the file resets the concentrations and sets ``alpha`` and ``beta``."""
    out = _replay("wofsy_goldstein")
    assert [r.label for r in out] == ["00", "01", "02", "10", "11", "12", "20", "21", "22"]
    for run in out:
        header, ref = _read(_DATA / f"wofsy_goldstein_{run.label}.scan")
        _close(run.scan_values, ref[:, 0])
        for j, column in enumerate(header[1:], start=1):
            _close([_column(r, column)[-1] for r in run.scan_results], ref[:, j])
