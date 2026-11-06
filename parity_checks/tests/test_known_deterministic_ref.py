"""An ODE DIFF is excused as KNOWN REF only while bngsim's run holds against the exact solution (GH #1548).

``l-type-calcium-channel-dynamics.bngl`` gives v_rec() = if(-70 + Voltage_Level < -20,
0.5, 0.05), and Voltage() relaxes as 50 - 49e^(-t): its steady state is the threshold
itself. On the exact solution Voltage() never reaches 50, v_rec() stays 0.5 and
Calcium() settles at 957.956. run_network's Voltage() lands on 50.0 at t = 25 (BNG
2.9.3, macOS arm64), v_rec() drops to 0.05 near t = 34, and Calcium() settles at
187.682: the other branch's steady state. bngsim's run stays one ulp under 50 and on
the exact solution, but by rounding, not by design (151 output points instead of 301
put it on run_network's branch), so the catalogue entry does not trust it: each run
is checked against the exact solution, and the DIFF scores when a check fails.

These tests pin that gate on trajectories built from the closed forms, its wiring into
the ODE worker, the matrix's rendering of it, and that the closed form is each
branch's steady state, without running an engine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import verdicts as V

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bng_parity"))

import bng_ode_run as bor  # noqa: E402
import generate_bng_matrix as gbm  # noqa: E402

STEM = "l-type-calcium-channel-dynamics"
RAW = (
    "Differs from run_network beyond tolerance at 1165/3010 points (10 species), full trajectory."
)
EXACT = bor.ltcc_steady_state(0.5)
OTHER = bor.ltcc_steady_state(0.05)
NAMES = list(EXACT)  # the .net's species, in its order
CA, V_ = NAMES.index("Calcium()"), NAMES.index("Voltage()")
CA_EXACT, CA_OTHER = EXACT["Calcium()"], OTHER["Calcium()"]
T = np.linspace(0.0, 150.0, 301)
REF_AT_50 = 50  # row of t = 25, where run_network's Voltage() reads 50.0
REF_LEAVES = 68  # row of t = 34, where its Calcium() leaves the exact branch


def _runs():
    """(bn, rn) shaped like the issue's runs: identical until run_network's
    Voltage() reaches 50 at t = 25; its Calcium() then falls to the other branch's
    steady state from t = 34, while bngsim's stays on the exact one."""
    x = np.tile(np.array([EXACT[n] for n in NAMES]), (len(T), 1))
    # The closed form rounds to 50.0 itself past t ~ 37; bngsim sits one ulp under.
    x[:, V_] = np.minimum(50.0 - 49.0 * np.exp(-T), np.nextafter(50.0, 0.0))
    x[:, CA] = CA_EXACT + (50.0 - CA_EXACT) * np.exp(-T)
    bn = x.copy()
    rn = x.copy()
    rn[REF_AT_50:, V_] = 50.0
    rn[REF_LEAVES:, CA] = CA_OTHER + (CA_EXACT - CA_OTHER) * np.exp(-(T[REF_LEAVES:] - 34.0))
    return (T.copy(), bn, list(NAMES)), (T.copy(), rn, [f"S{i + 1}" for i in range(len(NAMES))])


def _annotate(bn, rn, status="diff", stem=STEM, **kw):
    res = {"comment": RAW}
    return bor.annotate_known_ref(res, status, stem, bn, rn, atol=1e-8, **kw)


@pytest.mark.parametrize("v_rec", [0.5, 0.05])
def test_the_closed_form_is_the_steady_state_of_each_branch(v_rec):
    # Each balance read off the model's reaction rules, at Voltage() = 50 (v_open =
    # 5) and v_pka_shift() = -5, solved as a linear system rather than by the
    # closed form's algebra.
    ss = bor.ltcc_steady_state(v_rec)
    ca = ss["Calcium()"]
    c, a, r, i = 0.5, 5.0, v_rec, 5.0 * ca / (200.0 + ca)
    # Rows: CR, OR, OI balances, and the 500-channel total; columns CR, OR, OI, CI.
    m = np.array(
        [
            [-a, c, 0.0, r],
            [a, -(c + i), r, 0.0],
            [0.0, i, -(c + r), 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    cr, o_r, oi, ci = np.linalg.solve(m, [0.0, 0.0, 0.0, 500.0])
    assert c * oi - r * ci == pytest.approx(0.0, abs=1e-9)  # CI balances too
    assert 15.0 * o_r == pytest.approx(0.8 * ca, rel=1e-12)  # flux in = pump out
    assert ss["Voltage()"] == 50.0
    # p~ flips U -> P at 20 and back at 0.01, apart from everything else.
    for (g, s), n in {("C", "R"): cr, ("O", "R"): o_r, ("O", "I"): oi, ("C", "I"): ci}.items():
        pu = ss[f"LTCC(b,g~{g},loc~mem,p~U,s~{s})"]
        pp = ss[f"LTCC(b,g~{g},loc~mem,p~P,s~{s})"]
        assert pu + pp == pytest.approx(n, rel=1e-12)
        assert pp * 0.01 == pytest.approx(pu * 20.0, rel=1e-12)


def test_the_branches_are_the_two_engines_end_states():
    # bngsim's 957.9562053513 and run_network's 187.6822096489 on the issue's runs.
    assert pytest.approx(957.9562053513, rel=1e-10) == CA_EXACT
    assert pytest.approx(187.6822096489, rel=1e-10) == CA_OTHER
    assert bor.KNOWN_DETERMINISTIC_REF[STEM]["exact_settled"] == EXACT


def test_reference_leaving_the_exact_solution_is_excused_and_says_where():
    bn, rn = _runs()
    out = _annotate(bn, rn)
    assert out["subclass"] == "ref_bscb"
    note = V._note(out["comment"])
    assert note.startswith("Known reference divergence (GH #1548), not a bngsim error")
    assert "run_network's output reads Voltage() as 50 at t=25" in note
    assert "all 10 species within 1e-06 of their closed forms" in out["comment"]
    assert RAW in out["comment"]


def test_bngsim_on_the_threshold_at_one_output_scores():
    bn, rn = _runs()
    bn[1][200, V_] = 50.0
    out = _annotate(bn, rn)
    assert "subclass" not in out
    assert "bngsim's Voltage() reaches 50 at t=100" in V._note(out["comment"])
    assert RAW in out["comment"]


def test_bngsim_past_the_threshold_scores():
    bn, rn = _runs()
    bn[1][120:140, V_] = 50.0 + 3.5e-12
    assert "subclass" not in _annotate(bn, rn)


def test_bngsim_not_a_number_on_the_edge_scores():
    bn, rn = _runs()
    bn[1][30, V_] = np.nan
    out = _annotate(bn, rn)
    assert "subclass" not in out
    assert "reaches 50 at t=15" in V._note(out["comment"])


@pytest.mark.parametrize("value", [CA_OTHER, CA_EXACT * (1 + 2e-6), np.nan])
@pytest.mark.parametrize(("row", "t"), [(300, "150"), (150, "75"), (REF_AT_50, "25")])
def test_bngsim_calcium_off_the_closed_form_after_the_reference_leaves_scores(value, row, t):
    # Not just at the end: an excursion to the other branch that bngsim came back
    # from by t = 150 is still a wrong trajectory.
    bn, rn = _runs()
    bn[1][row, CA] = value
    out = _annotate(bn, rn)
    assert "subclass" not in out
    assert f"bngsim's Calcium() is {value:.9g} at t={t}, not the exact" in V._note(out["comment"])


def _move(bn, src, dst, share):
    v = bn[1]
    i, j = NAMES.index(src), NAMES.index(dst)
    moved = share * v[REF_AT_50 + 1 :, i]
    v[REF_AT_50 + 1 :, i] -= moved
    v[REF_AT_50 + 1 :, j] += moved


@pytest.mark.parametrize(
    "corrupt",
    [
        # Review of #1548: Calcium() exact, the rest of the state not.
        lambda bn: _move(bn, "LTCC(b,g~O,loc~mem,p~P,s~I)", "LTCC(b,g~O,loc~mem,p~U,s~I)", 0.4),
        lambda bn: _move(bn, "LTCC(b,g~C,loc~mem,p~P,s~I)", "LTCC(b,g~C,loc~mem,p~P,s~R)", 1.0),
        lambda bn: bn[1].__setitem__((slice(REF_AT_50 + 1, None), V_), 45.0),
    ],
    ids=["dephosphorylated", "recovered", "voltage-held-low"],
)
def test_bngsim_off_the_closed_form_in_any_species_scores(corrupt):
    bn, rn = _runs()
    corrupt(bn)
    out = _annotate(bn, rn)
    assert "subclass" not in out
    assert "at t=25.5, not the exact" in V._note(out["comment"])


def test_a_small_species_is_held_to_its_closed_form_relatively():
    # Review of #1548: the p~U states are 0.013 to 0.21, so an absolute bound would
    # let a few per cent through.
    bn, rn = _runs()
    j = NAMES.index("LTCC(b,g~C,loc~mem,p~U,s~R)")
    bn[1][REF_AT_50 + 5, j] += 1e-4  # 0.8% of 0.0131
    out = _annotate(bn, rn)
    assert "subclass" not in out
    assert "LTCC(b,g~C,loc~mem,p~U,s~R) is" in V._note(out["comment"])


def test_bngsim_within_the_closed_forms_tolerance_is_excused():
    bn, rn = _runs()
    bn[1][REF_AT_50:, :] *= 1 + 0.5 * bor.KNOWN_REF_SETTLED_REL
    bn[1][:, V_] = np.minimum(bn[1][:, V_], np.nextafter(50.0, 0.0))
    assert _annotate(bn, rn)["subclass"] == "ref_bscb"


def test_engines_that_differ_before_the_reference_reaches_the_threshold_score():
    bn, rn = _runs()
    rn[1][10:20, CA] *= 1.01
    out = _annotate(bn, rn)
    assert "subclass" not in out
    assert "already differ before run_network's Voltage() reads 50 at t=25" in V._note(
        out["comment"]
    )


def test_engines_that_differ_on_the_last_output_before_the_threshold_score():
    bn, rn = _runs()
    rn[1][REF_AT_50 - 1, CA] *= 1.1
    assert "already differ before" in _annotate(bn, rn)["comment"]


def test_the_prefix_verdict_is_given_the_runs_atol(monkeypatch):
    seen = []
    real = bor.differ.deterministic_verdict

    def spy(a, b, **kw):
        seen.append(kw.get("atol"))
        return real(a, b, **kw)

    monkeypatch.setattr(bor.differ, "deterministic_verdict", spy)
    bn, rn = _runs()
    assert _annotate(bn, rn)["subclass"] == "ref_bscb"
    assert seen == [1e-8]


def test_a_reference_that_never_reaches_the_threshold_scores():
    bn, rn = _runs()
    rn[1][:, V_] = bn[1][:, V_]
    out = _annotate(bn, rn)
    assert "subclass" not in out
    assert "run_network's Voltage() never reads 50" in V._note(out["comment"])


def test_a_reference_that_turns_not_a_number_has_not_reached_the_threshold():
    # Review of #1548: a NaN read as "reaches 50" would excuse the row with a false
    # account of it.
    bn, rn = _runs()
    rn[1][:, V_] = bn[1][:, V_]
    rn[1][60:, :] = np.nan
    out = _annotate(bn, rn)
    assert "subclass" not in out
    assert "never reads 50" in out["comment"]


def test_a_reference_that_starts_on_the_threshold_scores():
    bn, rn = _runs()
    rn[1][:, V_] = 50.0
    assert "starts at or past 50" in _annotate(bn, rn)["comment"]


@pytest.mark.parametrize("rows", [100, 300])
def test_runs_whose_outputs_differ_score(rows):
    # Review of #1548: bngsim's rows past run_network's last were never read.
    bn, rn = _runs()
    bn[1][120:, CA] = CA_OTHER  # off the exact solution past run_network's output
    rn = (rn[0][:rows], rn[1][:rows], rn[2])
    out = _annotate(bn, rn)
    assert "subclass" not in out
    assert "outputs differ" in V._note(out["comment"])


def test_runs_on_other_output_times_score():
    bn, rn = _runs()
    rn = (rn[0] * 1.001, rn[1], rn[2])
    assert "outputs differ" in _annotate(bn, rn)["comment"]


def test_a_run_without_the_edge_species_scores():
    (t, v, names), rn = _runs()
    names = [n if n != "Voltage()" else "V()" for n in names]
    out = _annotate((t, v, names), rn)
    assert "subclass" not in out
    assert "no species Voltage()" in out["comment"]


def test_names_that_do_not_cover_the_columns_score():
    (t, v, names), rn = _runs()
    out = _annotate((t, v, names[:-1]), rn)
    assert "subclass" not in out
    assert "names 9 species for 10 columns" in out["comment"]


def test_a_run_without_a_settled_species_scores():
    (t, v, names), rn = _runs()
    gone = "LTCC(b,g~C,loc~mem,p~U,s~R)"
    names = [n if n != gone else "X()" for n in names]
    assert f"no species {gone}" in _annotate((t, v, names), rn)["comment"]


@pytest.mark.parametrize(
    ("status", "stem", "kw"),
    [
        ("pass", STEM, {}),
        ("exception", STEM, {}),
        ("diff", "not_a_catalogued_model", {}),
        ("diff", STEM, {"multi_segment": True}),
    ],
)
def test_other_rows_are_left_alone(status, stem, kw):
    bn, rn = _runs()
    assert _annotate(bn, rn, status=status, stem=stem, **kw) == {"comment": RAW}


# ── the ODE worker calls the gate ──────────────────────────────────────────────
_BNGL = 'simulate({method=>"ode",t_end=>150,n_steps=>300,rtol=>1e-8,atol=>1e-8})\n'


class _Queue:
    def __init__(self):
        self.rows = []

    def put(self, row):
        self.rows.append(row)


def _worker_row(tmp_path, monkeypatch, stem=STEM):
    bn, rn = _runs()

    def generate_network(
        bngl_text, bng2_pl, workdir, timeout=None, gen_network=None, state_prefix=""
    ):
        out = Path(workdir) / "model.net"
        out.write_text("# faked\n")
        return out, 0.01, ""

    def bn_ode_net(net_path, t_start, t_end, n_points, rtol, atol, codegen=None):
        assert (t_start, t_end, n_points) == (0.0, 150.0, 301)
        return (*bn, {"config": {"codegen": False}})

    def run_network_ode(net_path, run_network_bin, **kw):
        return (*rn, {"engine": "faked run_network"})

    monkeypatch.setattr(bor.bc, "generate_network", generate_network)
    monkeypatch.setattr(bor.bc, "bn_ode_net", bn_ode_net)
    monkeypatch.setattr(bor.bc, "run_network_ode", run_network_ode)
    bngl = tmp_path / f"{stem}.bngl"
    bngl.write_text(_BNGL)
    spec = {
        "key": f"fast/x/{stem}.bngl:ode",
        "model_id": f"fast/x/{stem}.bngl",
        "method": "ode",
        "metric": "max_rel_err",
        "tol": 1e-4,
        "bngl": str(bngl),
        "rtol": 1e-8,
        "atol": 1e-8,
        "cap": 120.0,
        "sub_timeout": 120.0,
        "settings": None,
        "bng2_pl": None,
        "run_network_bin": None,
        "inject": None,
        "codegen": False,
    }
    q = _Queue()
    bor._worker(spec, q)
    return q.rows[0]


def test_the_ode_worker_tags_the_catalogued_divergence(tmp_path, monkeypatch):
    row = _worker_row(tmp_path, monkeypatch)
    assert (row["status"], row.get("subclass")) == ("diff", "ref_bscb")
    assert row["comment"].startswith("Known reference divergence (GH #1548)")


def test_the_ode_worker_leaves_an_uncatalogued_divergence_scoring(tmp_path, monkeypatch):
    row = _worker_row(tmp_path, monkeypatch, stem="some_other_model")
    assert (row["status"], row.get("subclass")) == ("diff", None)


def test_the_ode_worker_leaves_a_multi_segment_divergence_scoring(tmp_path, monkeypatch):
    # A carry-over job's runs are aligned by nearest time, not row for row, so the
    # gate does not read them, whatever they hold.
    bn, rn = _runs()

    class _Result:
        time, species, species_names = bn

    def representative_spec(text, track, net_params, atol, rtol):
        return {
            "t_start": 0.0,
            "t_end": 150.0,
            "n_steps": 300,
            "rtol": rtol,
            "atol": atol,
            "dirty_carryover": True,
            "steps": ["the protocol's segments"],
            "rep_index": 0,
            "rep_stmt_idx": 0,
        }

    monkeypatch.setattr(bor.bc, "read_net_parameters", lambda net_path: {})
    monkeypatch.setattr(bor.bc, "representative_spec", representative_spec)
    monkeypatch.setattr(bor.bc, "multi_segment_replay", lambda *a, **k: (_Result, {}))
    monkeypatch.setattr(bor.bc, "native_protocol_oracle", lambda *a, **k: rn)
    row = _worker_row(tmp_path, monkeypatch)
    assert row["timing"]["spec"]["mode"] == "multi_segment"
    assert (row["status"], row.get("subclass")) == ("diff", None)


# ── the ODE matrix ─────────────────────────────────────────────────────────────
def test_the_ode_matrix_renders_known_ref_as_triaged_with_bngsim_passing():
    assert gbm.classify_row("DIFF", "ref_bscb") == ("status-triaged", "KNOWN REF")
    assert gbm.classify_row("DIFF", None) == ("status-failed", "FAILED")
    assert gbm.classify_row("DIFF", "known_artifact") == ("status-known", "KNOWN ARTIFACT")
    assert gbm.verdict_badge("DIFF", "bngsim", "ref_bscb") == ("PASS", "verdict-pass")
    assert gbm.verdict_badge("DIFF", "run_network", "ref_bscb") == ("KNOWN REF", "verdict-known")
    assert gbm.verdict_badge("DIFF", "bngsim") == ("DIFF", "verdict-diff")


def test_the_ode_matrix_page_counts_and_badges_known_ref(tmp_path):
    rows = [
        {"model_id": "a.bngl", "outcome": "DIFF", "subclass": "ref_bscb", "timing": {}},
        {"model_id": "b.bngl", "outcome": "DIFF", "subclass": None, "timing": {}},
        {"model_id": "c.bngl", "outcome": "PASS", "subclass": None, "timing": {}},
    ]
    meta = {"regime": "ode", "tally": {"DIFF": 2, "PASS": 1}}  # as write_report records it
    report = tmp_path / "report_ode.json"
    report.write_text(json.dumps({"_meta": meta, "results": rows}))
    out = tmp_path / "m.html"
    gbm.generate_html(report, out)
    html = out.read_text()
    assert "✗ FAILED: 1<" in html
    assert "⚑ TRIAGED: 1<" in html
    assert "✓ PASSED: 1<" in html
    # The KNOWN REF row: bngsim's cell passes, run_network's carries KNOWN REF.
    row = html[html.index('<tr class="status-triaged"') :]
    row = row[: row.index("</tr>")]
    assert row.count('"verdict verdict-pass">PASS<') == 1
    assert row.count('"verdict verdict-known">KNOWN REF<') == 1
    assert "verdict-diff" not in row
