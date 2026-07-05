"""Lock generate_matrix.py's timing/config extraction (issue #135, T2.3).

These pure helpers turn a report's per-engine timing dicts and the run-level
``_meta["config"]`` block into the values the HTML renders. They carry the
fiddly bits the matrix depends on — the millisecond precision tiers, the
combined-phase fallback for engines that cannot separate parse/interpret(/JIT)
without an instrumented build, and back-compat for reports written before the
config block existed — so they get explicit oracles here.
"""

from __future__ import annotations

import generate_matrix as gm
import pytest


# --------------------------------------------------------------------------- #
# fmt_ms — millisecond formatting with precision tiers
# --------------------------------------------------------------------------- #
class TestFmtMs:
    def test_none_and_zero_render_as_dash(self):
        assert gm.fmt_ms(None) == "—"
        assert gm.fmt_ms(0) == "—"

    def test_precision_tiers(self):
        assert gm.fmt_ms(0.0005) == "0.50ms"  # <1ms → 2 decimals
        assert gm.fmt_ms(0.0034) == "3.4ms"  # <10ms → 1 decimal
        assert gm.fmt_ms(0.123) == "123ms"  # <1000ms → integer
        assert gm.fmt_ms(2.5) == "2.50s"  # >=1000ms → seconds


# --------------------------------------------------------------------------- #
# parse_phase_timing — per-phase resolution + combined-phase fallback
# --------------------------------------------------------------------------- #
class TestParsePhaseTiming:
    def test_bngsim_combined_parse_interpret_blanks_interpret(self):
        # bngsim reports parse_interpret_sec combined, codegen separate.
        t = {
            "io_sec": 0.001,
            "parse_interpret_sec": 0.4,
            "codegen_sec": 0.05,
            "integrate_sec": 0.03,
        }
        p = gm.parse_phase_timing(
            t, combined_key="parse_interpret_sec", combined_folds_codegen=False
        )
        assert p["parse"] == 0.4
        assert p["interpret"] is None  # folded into the combined value
        assert p["codegen"] == 0.05  # stays separate
        assert p["total"] == 0.001 + 0.4 + 0.05 + 0.03

    def test_bngsim_separate_phases_used_directly(self):
        # If an engine ever reports fine-grained phases, they win over the combined.
        t = {"parse_sec": 0.2, "interpret_sec": 0.1, "codegen_sec": 0.0, "integrate_sec": 0.05}
        p = gm.parse_phase_timing(
            t, combined_key="parse_interpret_sec", combined_folds_codegen=False
        )
        assert p["parse"] == 0.2
        assert p["interpret"] == 0.1
        assert p["total"] == 0.2 + 0.1 + 0.05

    def test_roadrunner_combined_folds_codegen(self):
        # RR cannot split parse/interpret/LLVM-JIT → one combined phase that folds
        # codegen; both interpret and codegen blank.
        t = {"io_sec": 0.0, "parse_interpret_codegen_sec": 0.6, "integrate_sec": 0.04}
        p = gm.parse_phase_timing(
            t, combined_key="parse_interpret_codegen_sec", combined_folds_codegen=True
        )
        assert p["parse"] == 0.6
        assert p["interpret"] is None
        assert p["codegen"] is None
        assert p["total"] == 0.6 + 0.04  # None sub-phases count as 0

    def test_empty_timing_is_all_zero(self):
        p = gm.parse_phase_timing(
            {}, combined_key="parse_interpret_sec", combined_folds_codegen=False
        )
        assert p["total"] == 0
        assert p["interpret"] is None  # fallback taken (parse==0, interpret==0)


# --------------------------------------------------------------------------- #
# format_run_config — _meta["config"] → header strings (T2.1/T2.2)
# --------------------------------------------------------------------------- #
class TestFormatRunConfig:
    def test_config_present_no_overrides(self):
        meta = {
            "config": {
                "combo": "auto",
                "rtol": 1e-9,
                "atol": 1e-12,
                "env": {"BNGSIM_CODEGEN_JIT": None, "BNGSIM_LAPACK_DENSE": None},
                "force_dense_linear_solver": False,
            }
        }
        rc = gm.format_run_config(meta)
        assert rc["combo"] == "auto"
        assert rc["tol"] == "rtol=1e-09, atol=1e-12"
        assert rc["overrides"] == "none (engine auto)"

    def test_config_present_with_overrides(self):
        meta = {
            "config": {
                "combo": "mir",
                "rtol": 1e-9,
                "atol": 1e-12,
                "env": {"BNGSIM_CODEGEN_JIT": "mir", "BNGSIM_LAPACK_DENSE": None},
                "force_dense_linear_solver": True,
            }
        }
        rc = gm.format_run_config(meta)
        assert rc["combo"] == "mir"
        assert "BNGSIM_CODEGEN_JIT=mir" in rc["overrides"]
        assert "force_dense_linear_solver=True" in rc["overrides"]
        assert "BNGSIM_LAPACK_DENSE" not in rc["overrides"]  # unset → omitted

    def test_back_compat_no_config_block(self):
        # A report written before T2.1 has no config block: never raises, sane
        # defaults.
        rc = gm.format_run_config({"suite": "rr_parity"})
        assert rc["combo"] == "auto"
        assert rc["tol"] == "—"
        assert rc["overrides"] == "none (engine auto)"


# --------------------------------------------------------------------------- #
# combo index aggregation (T3.2)
# --------------------------------------------------------------------------- #
class TestComboIndex:
    def _result(self, model_id, total_minus_io):
        # A row whose bngsim phases sum to total_minus_io (io 0 for simplicity).
        return {
            "model_id": model_id,
            "timing": {"bngsim": {"parse_interpret_sec": total_minus_io, "integrate_sec": 0.0}},
        }

    def test_bngsim_total_sec(self):
        assert gm.bngsim_total_sec(self._result("m", 0.25)) == 0.25
        # A failed/refused row carries no bngsim timing → None (excluded).
        assert gm.bngsim_total_sec({"model_id": "m"}) is None
        assert gm.bngsim_total_sec({"model_id": "m", "timing": {}}) is None

    def test_combo_from_report_prefers_meta_then_filename(self):
        from pathlib import Path

        assert gm.combo_from_report({"config": {"combo": "mir"}}, Path("report_ode.json")) == "mir"
        # No config block → parse the filename.
        assert gm.combo_from_report({}, Path("report_ode.json")) == "auto"
        assert gm.combo_from_report({}, Path("report_ode__lapack.json")) == "lapack"

    def test_aggregate_speedups_vs_auto(self):
        # auto totals 1.0+2.0=3.0 over {a,b}; the combo runs the same two in
        # 2.0+2.0=4.0 → 3.0/4.0 = 0.75× (slower). A model only auto timed (c) is
        # excluded from the ratio.
        combos = {
            "auto": {"totals": {"a": 1.0, "b": 2.0, "c": 5.0}},
            "slowcombo": {"totals": {"a": 2.0, "b": 2.0}},
        }
        sp = gm.aggregate_combo_speedups(combos)
        assert sp["auto"]["speedup"] is None  # baseline
        assert sp["slowcombo"]["n_common"] == 2  # a, b (not c)
        assert abs(sp["slowcombo"]["speedup"] - 0.75) < 1e-12

    def test_aggregate_speedups_no_common_models(self):
        combos = {"auto": {"totals": {"a": 1.0}}, "x": {"totals": {"b": 1.0}}}
        sp = gm.aggregate_combo_speedups(combos)
        assert sp["x"]["speedup"] is None
        assert sp["x"]["n_common"] == 0


# --------------------------------------------------------------------------- #
# SSA matrix (T4.1)
# --------------------------------------------------------------------------- #
class TestSsaMatrix:
    def test_row_class_honest_ssa_policy(self):
        # PASS green; a real regression red; an attributed-away DIFF yellow; a
        # too_slow TIMEOUT gray (coverage gap, not a failure).
        assert gm._ssa_row_class("PASS", None)[0] == "status-passed"
        # GH #190: a vacuous pass (nothing above the low-count floor) is a gray
        # coverage gap, NOT a green agreement — and its badge says NO SIGNAL.
        assert gm._ssa_row_class("PASS", "vacuous_lowcount") == ("status-refused", "NO SIGNAL")
        assert gm._ssa_verdict_badges("PASS", "vacuous_lowcount") == (
            "NO SIGNAL",
            "verdict-refused",
            "NO SIGNAL",
            "verdict-refused",
        )
        assert gm._ssa_verdict_badges("PASS", None) == ("PASS", "verdict-pass", "PASS", "verdict-pass")
        assert gm._ssa_row_class("DIFF", "bngsim_suspect")[0] == "status-failed"
        assert gm._ssa_row_class("DIFF", "ode_level")[0] == "status-failed"
        assert gm._ssa_row_class("DIFF", "rr_known")[0] == "status-triaged"
        assert gm._ssa_row_class("DIFF", "diff_not_bngsim")[0] == "status-triaged"
        # GH #190: partial (contracted) and extended (expanded) horizons are both
        # provisional yellow, with their own badges.
        assert gm._ssa_row_class("DIFF", "partial_horizon") == ("status-triaged", "PARTIAL")
        assert gm._ssa_row_class("DIFF", "extended_horizon") == ("status-triaged", "EXTENDED")
        assert gm._ssa_verdict_badges("DIFF", "extended_horizon")[0] == "EXTENDED"
        # GH #190: RR time-event deficit — yellow "RR LIMIT", bngsim PASS / RR limited.
        assert gm._ssa_row_class("DIFF", "rr_time_event") == ("status-triaged", "RR LIMIT")
        assert gm._ssa_verdict_badges("DIFF", "rr_time_event") == (
            "PASS",
            "verdict-pass",
            "RR LIMIT",
            "verdict-reffail",
        )
        # Unverified variant: same RR limit, bngsim NOT claimed PASS (mean≠ODE for
        # nonlinear/heavy-tailed models is unconfirmable, not a bngsim fault).
        assert gm._ssa_row_class("DIFF", "rr_time_event_unverified") == (
            "status-triaged",
            "RR LIMIT",
        )
        assert gm._ssa_verdict_badges("DIFF", "rr_time_event_unverified") == (
            "UNVERIFIED",
            "verdict-reffail",
            "RR LIMIT",
            "verdict-reffail",
        )
        assert gm._ssa_row_class("EXCEPTION", None)[0] == "status-failed"
        assert gm._ssa_row_class("REFERENCE_FAILED", None)[0] == "status-triaged"
        assert gm._ssa_row_class("TIMEOUT", None)[0] == "status-refused"

    def test_generate_ssa_html_renders_screen_fields(self, tmp_path):
        # A minimal SSA _core report (the ssa_attribution schema). The matrix
        # MIRRORS the ODE one — same title shape, summary boxes, z-gate metric in
        # plain English, the green/yellow/red/gray convention, and the footer
        # legend — and never leaks the wrong reference engine (run_network) or an
        # internal knob name (effort). A report with no timing still renders the
        # table + legend (timing is measurement-only).
        report = {
            "_meta": {
                "method": "ssa",
                "reference_engine": "roadrunner",
                "config": {
                    "corpus": "roundtrip",
                    "effort": "high",
                    "n_replicates": 30,
                    "mean_z_tol": 5.0,
                },
                "versions": {"bngsim": "9.9.9", "roadrunner": "2.9.2"},
                "outcome_tally": {"PASS": 1, "DIFF": 1, "TIMEOUT": 1},
                "subclass_breakdown": {"bngsim_suspect": 1},
                "n_not_expected": 1,
                "n_results": 3,
            },
            "results": [
                {
                    "model_id": "good",
                    "outcome": "PASS",
                    "metric": "mean_zscore",
                    "value": 2.6,
                    "tol": 5.0,
                    "subclass": None,
                    "wall_sec": 1.2,
                    "comment": "",
                },
                {
                    "model_id": "suspect",
                    "outcome": "DIFF",
                    "metric": "mean_zscore",
                    "value": 9.1,
                    "tol": 5.0,
                    "subclass": "bngsim_suspect",
                    "wall_sec": 2.0,
                    "comment": "worst cell",
                },
                {
                    "model_id": "slow",
                    "outcome": "TIMEOUT",
                    "value": None,
                    "tol": None,
                    "subclass": None,
                    "wall_sec": 20.0,
                    "comment": "",
                },
            ],
        }
        rpath = tmp_path / "ssa_report.json"
        rpath.write_text(__import__("json").dumps(report))
        out = tmp_path / "ssa.html"
        gm.generate_ssa_html(rpath, out)
        html = out.read_text()
        assert "BNGsim vs RoadRunner — SSA Parity Matrix" in html
        assert "RoadRunner 2.9.2" in html  # the reference engine + version, not run_network
        assert "run_network" not in html
        assert "effort:" not in html.lower()  # no internal knob name in the header
        assert "summary-passed" in html and "PASSED" in html  # ODE-style summary boxes
        assert "mean z-score" in html and "9.10" in html  # the agreement metric, plain English
        assert "status-failed" in html  # the bngsim_suspect DIFF colored red
        assert "Row Color Legend" in html  # the shared footer legend
        assert "How each model is checked" in html  # plain-English config


# --------------------------------------------------------------------------- #
# SSA timing display (issue #135) — pure helpers + the timing-aware renderer
# --------------------------------------------------------------------------- #
def _bn_timing():
    return {
        "io_sec": 0.0,
        "parse_sec": 0.003,
        "interpret_sec": 0.002,
        "codegen_sec": 0.15,
        "load_sec": 0.155,
        "n_rep": 30,
        "rep_median_sec": 0.00006,
        "rep_cold_sec": 0.0003,
        "rep_warm_median_sec": 0.00006,
        "ensemble_sec": 0.0021,
        "events_per_rep": 6250.0,
        "events_per_time": 62.5,
        "events_total": 187500,
        "config": {"method": "Gillespie SSA (exact)", "codegen": "cc", "cached": False},
    }


def _rr_timing():
    return {
        "io_sec": 0.0001,
        "parse_sec": 0.002,
        "interpret_sec": 0.0005,
        "codegen_sec": 0.07,
        "jit_sec": 0.069,
        "load_sec": 0.0726,
        "model_cache_hit": 0.0,
        "n_rep": 30,
        "rep_median_sec": 0.00005,
        "rep_cold_sec": 0.00013,
        "rep_warm_median_sec": 0.00005,
        "ensemble_sec": 0.0015,
        "config": {"method": "Gillespie SSA (exact)", "codegen": "LLVM JIT"},
    }


class TestSsaTiming:
    def test_geomean_is_symmetric_under_inversion(self):
        assert gm._ssa_geomean([2.0, 0.5]) == pytest.approx(1.0)
        assert gm._ssa_geomean([4.0, 4.0]) == pytest.approx(4.0)

    def test_geomean_ignores_nonpositive_and_empty(self):
        assert gm._ssa_geomean([0.0, -1.0]) is None
        assert gm._ssa_geomean([]) is None

    def test_timing_tiers_bngsim_load_and_cold_warm(self):
        tiers = gm.render_ssa_timing_tiers(
            {"warmup": {"bngsim_sec": 0.09}, "bngsim": _bn_timing()}, engine="bngsim"
        )
        assert "Per-process startup" in tiers and "Per-model load" in tiers
        assert "Per-replicate ensemble" in tiers
        assert "Cold → warm" in tiers and "Ensemble (30 reps)" in tiers
        # GH #190: bngsim reuses one Simulator across replicates (like RoadRunner),
        # so there is no per-replicate setup cell anymore.
        assert "Setup / rep" not in tiers
        assert "_bngsim_core import" in tiers
        # GH #190: the SSA path now codegens the propensity vector (cc -O3 .so);
        # the bngsim load tier shows it as "Codegen (cc)" (RoadRunner's is the
        # combined "Codegen + JIT", which is not a bngsim label).
        assert "Codegen (cc)" in tiers
        assert "Codegen + JIT" not in tiers

    def test_timing_tiers_roadrunner_codegen_jit_no_setup(self):
        tiers = gm.render_ssa_timing_tiers(
            {"warmup": {"roadrunner_sec": 0.04}, "roadrunner": _rr_timing()}, engine="roadrunner"
        )
        assert "Codegen + JIT" in tiers  # RoadRunner load is JIT-dominated
        assert "Setup / rep" not in tiers  # RoadRunner reuses one compiled model
        assert "Library import" in tiers

    def test_verdict_badges_encode_attribution(self):
        # A RoadRunner-side divergence: BNGsim PASS (tracks its own ODE), RR DIFF.
        assert gm._ssa_verdict_badges("DIFF", "diff_not_bngsim") == (
            "PASS",
            "verdict-pass",
            "DIFF",
            "verdict-diff",
        )
        # A bngsim-suspect divergence flags both engines.
        assert gm._ssa_verdict_badges("DIFF", "bngsim_suspect") == (
            "DIFF",
            "verdict-diff",
            "DIFF",
            "verdict-diff",
        )
        # RoadRunner refused; BNGsim ran.
        bn, _, rr, _ = gm._ssa_verdict_badges("REFERENCE_FAILED", None)
        assert bn == "PASS" and rr == "REF-FAIL"
        # BNGsim raised.
        assert gm._ssa_verdict_badges("EXCEPTION", None)[0] == "EXCEPTION"

    def test_resolve_ssa_sbml_missing_is_none(self, tmp_path):
        assert gm._resolve_ssa_sbml("NOPE", {"biomodels_sbml_dir": str(tmp_path)}) is None

    def test_timed_out_model_shows_probe_activity(self, tmp_path):
        # A too_slow (TIMEOUT) row carries only a probe-recovered events_per_time
        # (the worker was killed before the ensemble). The matrix must still show
        # its activity, flagged as a bounded-window probe estimate, and sort it as
        # high activity rather than blank/last.
        import json

        report = {
            "_meta": {
                "method": "ssa",
                "reference_engine": "roadrunner",
                "config": {"corpus": "all", "n_replicates": 30, "mean_z_tol": 5.0},
                "versions": {"bngsim": "9.9.9", "roadrunner": "2.9.2"},
                "outcome_tally": {"PASS": 1, "TIMEOUT": 1},
                "n_results": 2,
            },
            "results": [
                {
                    "model_id": "calm",
                    "outcome": "PASS",
                    "metric": "mean_zscore",
                    "value": 1.0,
                    "tol": 5.0,
                    "subclass": None,
                    "wall_sec": 1.0,
                    "comment": "",
                    "timing": {"bngsim": _bn_timing(), "roadrunner": _rr_timing()},
                },
                {
                    "model_id": "wild",
                    "outcome": "TIMEOUT",
                    "value": None,
                    "tol": None,
                    "subclass": None,
                    "wall_sec": 40.0,
                    "comment": "",
                    "timing": {
                        "bngsim": {
                            "events_per_time": 42273.6,
                            "events_per_rep": 4227360.0,
                            "events_probe": True,
                        }
                    },
                },
            ],
        }
        rpath = tmp_path / "ssa_report.json"
        rpath.write_text(json.dumps(report))
        out = tmp_path / "ssa.html"
        gm.generate_ssa_html(rpath, out)
        html = out.read_text()
        assert "42,274" in html and "reaction events / time unit" in html
        assert "bounded-window probe" in html
        # The wild (timed-out, high-activity) model sorts AFTER the calm one.
        assert html.index("wild") > html.index("calm")

    def test_generate_ssa_html_full_mirror_of_ode(self, tmp_path):
        # The timing-bearing render must produce the same VISUAL STRUCTURE as the
        # ODE matrix: summary boxes, the top cost plots + wins tables, the three
        # timing tiers in each engine cell, and the shared footer legend.
        import json

        wu = {"bngsim_sec": 0.07, "roadrunner_sec": 0.04, "roadrunner_source": "engine"}
        report = {
            "_meta": {
                "method": "ssa",
                "reference_engine": "roadrunner",
                "config": {
                    "corpus": "all",
                    "effort": "high",
                    "n_replicates": 30,
                    "mean_z_tol": 5.0,
                    "min_mean_count": 2.0,
                    "biomodels_t_end": 100.0,
                    "biomodels_n_steps": 10,
                    "seed_base": 2000,
                    "per_model_timeout_sec": 40.0,
                    "jobs": 6,
                },
                "versions": {"bngsim": "9.9.9", "roadrunner": "2.9.2"},
                "machine": {"platform": "macOS-14", "processor": "arm", "cpu_count": 6},
                "outcome_tally": {"PASS": 1, "TIMEOUT": 1},
                "subclass_breakdown": {},
                "n_not_expected": 0,
                "n_results": 2,
            },
            "results": [
                {
                    "model_id": "good",
                    "outcome": "PASS",
                    "metric": "mean_zscore",
                    "value": 2.0,
                    "tol": 5.0,
                    "subclass": None,
                    "wall_sec": 1.0,
                    "comment": "",
                    "timing": {"warmup": wu, "bngsim": _bn_timing(), "roadrunner": _rr_timing()},
                },
                {
                    "model_id": "slow",
                    "outcome": "TIMEOUT",
                    "value": None,
                    "tol": None,
                    "subclass": None,
                    "wall_sec": 40.0,
                    "comment": "",
                    "timing": None,
                },
            ],
        }
        rpath = tmp_path / "ssa_report.json"
        rpath.write_text(json.dumps(report))
        out = tmp_path / "ssa.html"
        gm.generate_ssa_html(rpath, out)
        html = out.read_text()
        # Same three-column header as the ODE matrix.
        assert ">BNGsim</th>" in html and ">RoadRunner</th>" in html
        # Top-of-page plots + wins tables (mirrors the ODE cost plots).
        assert "<canvas" in html and 'id="cv_rep"' in html and 'id="cv_load"' in html
        assert "wins by reaction-rate bin" in html  # event-rate binning when events present
        # The three timing tiers inside the cells.
        assert "Per-process startup" in html and "Per-replicate ensemble" in html
        # Shared footer legend, with the SSA z-gate explanation.
        assert "Row Color Legend" in html
        assert "How agreement (PASS vs DIFF) is decided" in html
        # The TIMEOUT row carries no timing — it must render (gray), not crash.
        assert "TOO SLOW" in html
        # Stochastic-activity ordering: the model cell shows reaction-events/time,
        # and the plot/sort axis is activity (not species).
        assert "reaction events / time unit" in html
        assert "ordered by stochastic activity" in html

    def test_ssa_html_falls_back_to_species_without_events(self, tmp_path):
        # A report whose timing has no events field orders by species count, with
        # the species axis label (back-compat for pre-events reports).
        import json

        bn = _bn_timing()
        rr = _rr_timing()
        bn.pop("events_per_time")
        bn.pop("events_per_rep")
        wu = {"bngsim_sec": 0.07, "roadrunner_sec": 0.04, "roadrunner_source": "engine"}
        report = {
            "_meta": {
                "method": "ssa",
                "reference_engine": "roadrunner",
                "config": {"corpus": "all", "n_replicates": 30, "mean_z_tol": 5.0, "jobs": 6},
                "versions": {"bngsim": "9.9.9", "roadrunner": "2.9.2"},
                "outcome_tally": {"PASS": 1},
                "n_results": 1,
            },
            "results": [
                {
                    "model_id": "good",
                    "outcome": "PASS",
                    "metric": "mean_zscore",
                    "value": 2.0,
                    "tol": 5.0,
                    "subclass": None,
                    "wall_sec": 1.0,
                    "comment": "",
                    "timing": {"warmup": wu, "bngsim": bn, "roadrunner": rr},
                },
            ],
        }
        rpath = tmp_path / "ssa_report.json"
        rpath.write_text(json.dumps(report))
        out = tmp_path / "ssa.html"
        gm.generate_ssa_html(rpath, out)
        html = out.read_text()
        assert "ordered by complexity" in html  # species fallback
        assert "reaction events / time unit" not in html  # no activity line
