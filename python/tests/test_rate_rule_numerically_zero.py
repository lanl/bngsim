"""A rate-rule target a hair below an exact zero, on the non-finite retry (issue #1340).

MODEL1812040005's P is a rate-rule target at an exact equilibrium of 0, read by
``a_hat = a*P^z`` with z fractional. Summed in the order its species are
written, its derivative is exactly 0 and P stays 0; reversed, the same sums
leave -3.6e-27, where ``P^z`` is NaN, and the run failed
(``CV_REPTD_RHSFUNC_ERR``). The value RHS's retry on a clamped state left a
rate-rule target alone (#706: it may be negative by design, a membrane potential
at -61). A slot negative by less than its own atol is now read as 0 on that
retry, as the sensitivity RHS reads one (GH #392); one negative by more is left
alone, and still fails where its rate law is undefined.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bngsim
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source_root import bngsim_source_root  # noqa: E402

_ROOT = bngsim_source_root() or Path(__file__).resolve().parents[2]
_CORPUS = (
    _ROOT / "parity_checks" / "rr_parity" / "models" / "MODEL1812040005" / "MODEL1812040005.xml"
)

_RUN = {"t_span": (0, 10), "n_points": 11, "rtol": 1e-8, "atol": 1e-9}


def _sinking(c: str, order: str, power: str = "0.5") -> bngsim.Model:
    """P' = -c from P = 0, read by P^power in A's decay: for c = 1e-30, P sinks
    to -1e-29 by t = 10, inside atol; for c = 1e-3, to -1e-2, outside it."""
    decl = "A = 1, P = 0" if order == "AP" else "P = 0, A = 1"
    return bngsim.Model.from_antimony_string(
        f"model m\n  species {decl};\n  k = 0.1; c = {c};\n  P' = -c;\n"
        f"  J: A -> ; k*A*P^{power};\nend"
    )


def _run(model):
    r = bngsim.Simulator(model, method="ode").run(**_RUN)
    names = list(r.species_names)
    return {n: np.asarray(r.species)[:, names.index(n)] for n in names}


@pytest.mark.parametrize("order", ["AP", "PA"])
def test_a_rate_rule_target_inside_atol_of_zero_is_read_as_zero(order):
    x = _run(_sinking("1e-30", order))
    assert x["P"][-1] == pytest.approx(-1e-29)  # the state itself is not moved
    np.testing.assert_array_equal(x["A"], 1.0)  # sqrt(P) read at P = 0


def test_a_run_with_sensitivities_reads_it_the_same_way():
    """A run with sensitivity columns sets its solver up on another path than a
    plain one (create_cvode_core, not the warm cache), which fills the
    tolerance the retry reads as well. P^1.5, flat at 0: sqrt's slope there is
    infinite, which no reading of P makes a sensitivity of."""
    model = _sinking("1e-30", "AP", power="1.5")
    r = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(**_RUN)
    names = list(r.species_names)
    np.testing.assert_array_equal(np.asarray(r.species)[:, names.index("A")], 1.0)


def test_the_answer_does_not_depend_on_the_order_species_are_written():
    a, b = _run(_sinking("1e-30", "AP")), _run(_sinking("1e-30", "PA"))
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])


@pytest.mark.parametrize("order", ["AP", "PA"])
def test_a_rate_rule_target_negative_past_atol_is_left_alone(order):
    """The control: past atol it is not numerically zero, and sqrt of it is not
    read as anything. The run fails as it did."""
    with pytest.raises(bngsim.SimulationError):
        _run(_sinking("1e-3", order))


def test_model1812040005_runs_in_either_order(tmp_path):
    if not _CORPUS.is_file():
        pytest.skip(f"test data not found: {_CORPUS}")
    libsbml = pytest.importorskip("libsbml")
    doc = libsbml.readSBMLFromFile(str(_CORPUS))
    species = doc.getModel().getListOfSpecies()
    items = [species.remove(0) for _ in range(species.size())]
    for s in reversed(items):
        species.appendAndOwn(s)
    reversed_path = tmp_path / "reversed.xml"
    libsbml.writeSBMLToFile(doc, str(reversed_path))
    run = {"t_span": (0, 1), "n_points": 101, "rtol": 1e-6, "atol": 1e-9}
    out = []
    for path in (_CORPUS, reversed_path):
        r = bngsim.Simulator(bngsim.Model.from_sbml(str(path)), method="ode").run(**run)
        names = list(r.species_names)
        out.append({n: np.asarray(r.species)[:, names.index(n)] for n in names})
    # Each species within 1e-8 of its own peak, and P (whose peak is 0) within
    # the run's atol: what the two orders leave apart is rounding.
    for name, x in out[0].items():
        bound = 1e-8 * float(np.abs(x).max()) + run["atol"]
        assert np.abs(x - out[1][name]).max() <= bound, name
