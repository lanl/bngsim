"""Issue #881: which fallback oracle scores an SSA row must not depend on runner speed.

When run_network's SSA fails, bng_stoch_run scores bngsim against net_gillespie, else
libRoadRunner. net_gillespie used to give up at a 90 s wall budget, so the same code
and seeds were scored by net_gillespie on a fast runner and by RoadRunner on a slow
one, and ``v08`` flipped PASS/DIFF between nights. Its support is now decided by its
gate and deterministic event caps, the wall clock is a safety stop that leaves the
row unscored and says so, and every row names why net_gillespie produced nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bng_parity"))

import bng_stoch_run as bsr  # noqa: E402
import net_gillespie as ng  # noqa: E402

_NET = """begin parameters
    1 k {k}
end parameters
begin species
    1 A() {a0}
end species
begin reactions
    1 {reaction} {rate}
end reactions
begin groups
    1 Atot 1
end groups
"""


def _net(tmp_path, *, k="1", a0="100", reaction="1 0", rate="k") -> str:
    path = tmp_path / "m.net"
    path.write_text(_NET.format(k=k, a0=a0, reaction=reaction, rate=rate))
    return str(path)


# ── net_gillespie's outcome ──────────────────────────────────────────────────


def test_a_supported_net_runs_and_counts_its_events(tmp_path):
    out = ng.run_net_gillespie(_net(tmp_path), np.linspace(0, 1, 5), n_rep=3, seed_base=1)
    assert out.reason == "ok"
    t, vals, names = out.values
    assert vals.shape == (3, 5, 1) and names == ["Atot"]
    assert out.events > 0


def test_the_event_count_is_the_same_every_run(tmp_path):
    # What makes the oracle choice deterministic: fixed seeds, fixed events.
    net = _net(tmp_path)
    a = ng.run_net_gillespie(net, np.linspace(0, 1, 5), n_rep=3, seed_base=1)
    b = ng.run_net_gillespie(net, np.linspace(0, 1, 5), n_rep=3, seed_base=1)
    assert a.events == b.events


def test_an_unsupported_net_says_why(tmp_path):
    out = ng.run_net_gillespie(
        _net(tmp_path, rate="Atot"), np.linspace(0, 1, 5), n_rep=2, seed_base=1
    )
    assert out.values is None and out.reason == "unsupported"
    assert "rate 'Atot' is not a constant" in out.describe()


def test_the_ensemble_event_cap_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(ng, "MAX_ENSEMBLE_EVENTS", 50)
    out = ng.run_net_gillespie(_net(tmp_path), np.linspace(0, 1, 5), n_rep=3, seed_base=1)
    assert out.values is None and out.reason == "event_cap"
    assert "MAX_ENSEMBLE_EVENTS = 50" in out.describe()


def test_the_ensemble_cap_bounds_cost_at_every_network_size():
    # An event scans the reactions, so on a 1,500-reaction network it costs about
    # 14x what it does on a 10-reaction one. A flat event cap admitted ~1,000 s of
    # work there, which a fast runner finishes and a slow one stops. The cap is
    # scaled so the worst admitted ensemble costs the same at any size.
    assert ng.ensemble_event_cap(10) == ng.MAX_ENSEMBLE_EVENTS
    budget = ng.MAX_ENSEMBLE_EVENTS * (ng._EVENT_FIXED_COST + 10)
    for nr in (1, 10, 100, 500, ng.MAX_REACTIONS):
        assert ng.ensemble_event_cap(nr) * (ng._EVENT_FIXED_COST + nr) <= budget
    assert ng.ensemble_event_cap(ng.MAX_REACTIONS) < ng.MAX_ENSEMBLE_EVENTS // 10


def test_the_wall_budget_is_reported_as_the_safety_stop(tmp_path):
    # 0 -> A at 1e5/s for 1 s passes the 65,536-event deadline check once.
    net = _net(tmp_path, k="1e5", a0="0", reaction="0 1")
    out = ng.run_net_gillespie(net, np.linspace(0, 1, 5), n_rep=1, seed_base=1, wall_budget_sec=0)
    assert out.values is None and out.reason == "wall_budget"
    assert "safety stop" in out.describe()


def test_an_oracle_exception_is_captured_not_raised(tmp_path):
    out = ng.run_net_gillespie(tmp_path / "missing.net", np.linspace(0, 1, 5), 2, 1)
    assert out.values is None and out.reason == "error"
    assert "FileNotFoundError" in out.describe()


def test_the_old_entry_point_still_returns_the_ensemble_or_none(tmp_path):
    grid = np.linspace(0, 1, 5)
    assert ng.net_gillespie_ensemble(_net(tmp_path), grid, 2, 1) is not None
    assert ng.net_gillespie_ensemble(_net(tmp_path, rate="Atot"), grid, 2, 1) is None


# ── The row ──────────────────────────────────────────────────────────────────


def _ensemble(seed=0):
    t = np.linspace(0, 1, 5)
    vals = np.random.default_rng(seed).normal(100, 1, (10, 5, 1))
    return (t, vals, ["Atot"])


def _score(ng_outcome, rr_oracle):
    res = {}
    bn = _ensemble()
    bsr._score_fallback_oracles(
        res, "ssa", "run_network", "edgepop crash", bn, ng_outcome, bn, rr_oracle, None
    )
    return res


def test_net_gillespie_scores_the_row_when_it_ran():
    res = _score(ng.NetGillespieOutcome(_ensemble(1), "ok"), _ensemble(2))
    assert res["subclass"] == "oracle_net_gillespie"
    assert res["status"] in ("pass", "diff")


def test_a_safety_stop_leaves_the_row_unscored_rather_than_switching_oracle():
    stopped = ng.NetGillespieOutcome(None, "wall_budget", "900 s", 912.0, 4_000_000)
    res = _score(stopped, _ensemble(2))  # RoadRunner's ensemble is there, and unused
    assert res["status"] == "reference_failed"
    assert res["subclass"] == "oracle_net_gillespie_budget"
    assert "900 s safety stop after 912 s" in res["comment"]
    assert "issue #881" in res["comment"]


@pytest.mark.parametrize(
    "outcome, says",
    [
        (ng.NetGillespieOutcome(None, "unsupported", "rate 'f' is not a constant"), "support"),
        (ng.NetGillespieOutcome(None, "event_cap", "MAX_ENSEMBLE_EVENTS = 20000000"), "cap"),
        (ng.NetGillespieOutcome(None, "error", "ValueError: x"), "raised"),
    ],
    ids=["unsupported", "event_cap", "error"],
)
def test_roadrunner_scores_a_deterministic_refusal_and_the_row_says_why(outcome, says):
    res = _score(outcome, _ensemble(2))
    assert res["subclass"] == "oracle_roadrunner"
    assert outcome.describe() in res["comment"] and says in res["comment"]
    assert "did not apply" not in res["comment"]


def test_with_no_oracle_the_row_is_unscored_and_says_why():
    res = _score(ng.NetGillespieOutcome(None, "unsupported", "rate 'f' is not a constant"), None)
    assert res["status"] == "reference_failed"
    assert "does not support this .net (rate 'f' is not a constant)" in res["comment"]
