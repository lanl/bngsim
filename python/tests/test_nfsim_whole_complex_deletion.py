"""Issue #1525: NFsim deletes a whole complex once per complex, not once per match.

A rule that deletes a whole species (no DeleteMolecules) removes the complex the
match sits in, and BNG2.pl's network gives it one reaction per complex at the
rule's rate, however many matches the complex holds. NFsim's reactant list holds
one entry per match, so ``method="nf"`` fired the rule once per match: a homodimer
at 2k, a 4-ring at 4k, the asymmetric M.N.M under ``M() -> 0`` at 2k, with no
error. NFsim now fires such a pick with probability 1/m, m the matches in the
picked complex. MatchOnce asks for the same counting, and NFsim's own per-complex
count for it never took effect, so a MatchOnce reactant is thinned the same way.

``tests/data/nfsim/whole_complex_deletion.bngl`` holds the pools: X0 = 2000
complexes each, k = 1e-3, so the count left at t = 500 is Binomial(X0, exp(-k t)),
mean 1213.1. Each test averages three seeds; the bands are 5 standard errors
(12.6 complexes), and the rate before the fix (2k: 735.8, 4k: 270.7) is 38 and
more standard errors outside them.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path

import bngsim
import pytest
from bngsim import ModelError, NfsimSession

pytestmark = pytest.mark.skipif(
    not getattr(bngsim, "HAS_NFSIM", False), reason="NFsim not compiled in"
)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim"
NET = DATA / "whole_complex_deletion.net"
XML = DATA / "whole_complex_deletion.xml"

X0, K, KT, T_END = 2000.0, 1e-3, 2.0, 500.0
SEEDS = (1, 2, 3)
SURVIVE = math.exp(-K * T_END)
EXACT = X0 * SURVIVE  # 1213.1
SE = math.sqrt(X0 * SURVIVE * (1 - SURVIVE) / len(SEEDS))

# observable -> (molecules per complex, the complexes left by BNG2.pl's network)
POOLS = {
    "A_m": (2, EXACT),  # homodimer A(d!1).A(d!1) -> 0
    "R_m": (4, EXACT),  # 4-ring
    "N_m": (1, EXACT),  # M() -> 0 on asymmetric M(s~a).N.M(s~b)
    "Qbound": (2, EXACT),  # Q() -> 0 on a dimer, beside monomers
    "G_m": (2, EXACT),  # homodimer, global function rate
    "H_m": (2, EXACT),  # homodimer + catalyst X, bimolecular
    "K_m": (1, X0 * math.exp(-3 * K * T_END)),  # local function, 3k for this species
    "J_m": (2, EXACT),  # {MatchOnce} homodimer
}
# Already right before the fix: one match per complex, DeleteMolecules (which
# BNG2.pl writes with symmetry factor 0.5), and a total rate.
CONTROLS = {
    "Qfree": (1, EXACT),  # Q() -> 0 on monomers
    "C_m": (1, EXACT),  # heterodimer
    "B_m": (2, EXACT),  # homodimer with DeleteMolecules
    "Z_m": (1, EXACT),  # Z(s!1,s!2).P(z!1).P(z!2), written centre first
}


def _band(name: str, expected: float) -> float:
    if name == "K_m":
        q = expected / X0
        return 5 * math.sqrt(X0 * q * (1 - q) / len(SEEDS))
    return 5 * SE


def _complexes(observables, per: int) -> float:
    return float(observables[-1]) / per


@pytest.fixture(scope="module", params=[True, False], ids=["complexes-tracked", "untracked"])
def simulator_means(request) -> dict[str, float]:
    """Pool means over SEEDS from Simulator(method='nf'). With
    block_same_complex_binding=False and only Molecules observables, NFsim tracks
    no complexes, and the count of matches in a complex comes from its bonds."""
    runs: dict[str, list[float]] = {}
    for seed in SEEDS:
        sim = bngsim.Simulator(
            bngsim.Model.from_net(str(NET)),
            method="nf",
            xml_path=str(XML),
            block_same_complex_binding=request.param,
        )
        r = sim.run(t_span=(0, T_END), n_points=2, seed=seed)
        for name, (per, _) in {**POOLS, **CONTROLS, "T_m": (2, 0.0)}.items():
            runs.setdefault(name, []).append(_complexes(r.observables[name], per))
    return {name: statistics.fmean(v) for name, v in runs.items()}


@pytest.mark.parametrize("name", list(POOLS))
def test_a_complex_is_deleted_once_at_the_rules_rate(simulator_means, name):
    """Before the fix every pool here left 2k's or 4k's count (735.8, 270.7)."""
    per, expected = POOLS[name]
    assert simulator_means[name] == pytest.approx(expected, abs=_band(name, expected))


@pytest.mark.parametrize("name", list(CONTROLS))
def test_a_complex_with_one_match_is_deleted_as_before(simulator_means, name):
    """Control: one match per complex, or DeleteMolecules' symmetry factor."""
    per, expected = CONTROLS[name]
    assert simulator_means[name] == pytest.approx(expected, abs=_band(name, expected))


def test_a_total_rate_is_not_thinned(simulator_means):
    """Control: a TotalRate rule's rate is the rule's whole rate, kT = 2 complexes
    per unit time, so 1000 of the 2000 are gone at t = 500 (Poisson, sd 18.3 over
    three seeds)."""
    assert simulator_means["T_m"] == pytest.approx(
        X0 - KT * T_END, abs=5 * math.sqrt(KT * T_END / 3)
    )


def test_the_network_bng2pl_generates_runs_each_complex_at_k():
    """Control: the reference. BNG2.pl's network for the same rules gives every
    complex one reaction at k (the local-function species 3k), so its ODE run
    leaves X0*exp(-k*t) of each."""
    r = bngsim.Simulator(bngsim.Model.from_net(str(NET)), method="ode").run(
        t_span=(0, T_END), n_points=2
    )
    for name, (per, expected) in {**POOLS, **CONTROLS}.items():
        assert _complexes(r.observables[name], per) == pytest.approx(expected, rel=1e-6), name


def test_an_nfsim_session_deletes_a_complex_once():
    """The session path showed the same 2k and 4k."""
    runs: dict[str, list[float]] = {}
    for seed in SEEDS:
        with NfsimSession(str(XML), molecule_limit=1_000_000) as nf:
            nf.initialize(seed=seed)
            nf.simulate(0.0, T_END, 2)
            values = dict(zip(nf.get_observable_names(), nf.get_observable_values(), strict=False))
        for name in ("A_m", "R_m", "N_m"):
            runs.setdefault(name, []).append(values[name] / POOLS[name][0])
    for name, values in runs.items():
        assert statistics.fmean(values) == pytest.approx(EXACT, abs=5 * SE), name


# ─── MatchOnce on a rule that changes a state ───────────────────────────────


def test_match_once_counts_a_complex_once_on_a_state_change():
    """Jmo ({MatchOnce}) flips one subunit of a homodimer at k per complex, as
    BNG2.pl's network has it, and Wpl (no MatchOnce) at 2k. Jmo ran at 2k too:
    1264 flipped at t = 500 where 786.9 is the mean."""
    net, xml = DATA / "match_once_state_change.net", DATA / "match_once_state_change.xml"
    flipped: dict[str, list[float]] = {}
    for seed in SEEDS:
        sim = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="nf", xml_path=str(xml))
        r = sim.run(t_span=(0, T_END), n_points=2, seed=seed)
        for name in ("J1", "W1"):
            flipped.setdefault(name, []).append(float(r.observables[name][-1]))
    once, twice = X0 * (1 - SURVIVE), X0 * (1 - SURVIVE**2)
    assert statistics.fmean(flipped["J1"]) == pytest.approx(once, abs=5 * SE)
    assert statistics.fmean(flipped["W1"]) == pytest.approx(twice, abs=5 * SE)


@pytest.mark.parametrize("tracked", [True, False], ids=["complexes-tracked", "untracked"])
def test_match_once_on_a_context_reactant_is_not_counted_down_twice(tracked):
    """Uctx makes Z at k per U homodimer, the dimer pure context, so Z = k*X0*t =
    1000 at t = 500. NFsim counts a context reactant once per complex already
    while it tracks complexes, and thinning it as well would halve Z: a control
    there. Untracked, main counted it per match (2000), and the thinning is what
    counts it once."""
    net, xml = DATA / "match_once_state_change.net", DATA / "match_once_state_change.xml"
    made = []
    for seed in SEEDS:
        sim = bngsim.Simulator(
            bngsim.Model.from_net(str(net)),
            method="nf",
            xml_path=str(xml),
            block_same_complex_binding=tracked,
        )
        made.append(float(sim.run(t_span=(0, T_END), n_points=2, seed=seed).observables["Zt"][-1]))
    expected = K * X0 * T_END
    assert statistics.fmean(made) == pytest.approx(
        expected, abs=5 * math.sqrt(expected / len(SEEDS))
    )


# ─── Michaelis-Menten ────────────────────────────────────────────────────────


def test_a_michaelis_menten_rule_counted_per_complex_is_refused():
    """The substrate total counts a dimer twice and the law is not linear in it,
    so no thinning corrects it: 417.5 dimers were left at t = 500 where BNG2.pl's
    network leaves 629.7. SS_mm deletes the whole dimer and Tmo_mm's substrate
    carries MatchOnce; both are refused, by name. Q_mm, whose substrate is one
    molecule with no components, is not named."""
    net, xml = DATA / "mm_whole_complex_deletion.net", DATA / "mm_whole_complex_deletion.xml"
    with pytest.raises(
        ModelError, match=r"rules 'SS_mm', 'Tmo_mm'\. NFsim counts each match"
    ) as exc:
        bngsim.Simulator(bngsim.Model.from_net(str(net)), method="nf", xml_path=str(xml))
    assert "Q_mm" not in str(exc.value)
    with pytest.raises(ModelError, match="issue #1525"):
        NfsimSession(str(xml))


def test_a_michaelis_menten_monomer_deletion_runs():
    """Control: a substrate of one molecule with no components is always a
    complex of its own."""
    net, xml = DATA / "mm_monomer_deletion.net", DATA / "mm_monomer_deletion.xml"
    sim = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="nf", xml_path=str(xml))
    r = sim.run(t_span=(0, 10), n_points=2, seed=1)
    assert float(r.observables["Qt"][-1]) <= 100.0
