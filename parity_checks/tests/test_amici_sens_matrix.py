"""Lock the amici_parity forward-sensitivity matrix's vacuous-pass rendering.

Issue #328: a sensitivity comparison can be reported PASS when the WHOLE tensor
lies below the magnitude either solver can resolve — nothing was meaningfully
compared, yet the row is indistinguishable from a real pass. The runner already
records the witnesses (``n_resolvable_params``, ``max_abs_sx``, ``state_span``)
and notes ``DEGENERATE`` in the row comment; the missing piece of the fix's
option (1) was a *distinct badge in the matrix* so a reader can see the row
established nothing. These tests pin that rendering.

The treatment mirrors the SSA matrix's vacuous-pass handling (GH #190): the
outcome stays PASS in the report and the tally, but the row renders as a gray
coverage gap with a ``NO SIGNAL`` badge instead of a green ``PASS``. That policy
lives in one classifier (``sens_row_class``), so a change to it is a visible edit
here rather than a silent shift in what the page shows.
"""

from __future__ import annotations

import json

import generate_amici_sens_matrix as gsm


class TestSensRowClass:
    """The one classifier that decides how a row reads — pinned directly so the
    end-to-end render test below can trust the pieces it composes."""

    def test_a_zero_resolvable_pass_is_a_gray_no_signal_row(self):
        # The whole point of #328: a PASS with no resolvable parameter column is
        # NOT a green agreement. Gray coverage-gap styling, distinct badge.
        assert gsm.sens_row_class("PASS", {"n_resolvable_params": 0}) == (
            "status-refused",
            "NO SIGNAL",
        )

    def test_a_resolvable_pass_stays_a_green_pass(self):
        assert gsm.sens_row_class("PASS", {"n_resolvable_params": 3}) == (
            "status-passed",
            "PASS",
        )

    def test_an_unassessed_floor_is_not_a_degeneracy_claim(self):
        # None means the floor could not be assessed (no parameter values), which
        # is not the same as "assessed and found vacuous". It must read as an
        # ordinary PASS, never NO SIGNAL — the field must not claim a verdict the
        # run had no basis for.
        assert gsm.sens_row_class("PASS", {"n_resolvable_params": None}) == (
            "status-passed",
            "PASS",
        )
        # And a row with the field entirely absent (a report predating #330).
        assert gsm.sens_row_class("PASS", {}) == ("status-passed", "PASS")

    def test_a_diff_is_never_no_signal_even_at_zero_resolvable(self):
        # A DIFF failed on something real — the noise floor never forgives a
        # one-sided non-finite cell, so a NaN column still DIFFs even when every
        # column is otherwise sub-floor. Degeneracy is a PASS-only qualifier.
        assert gsm.sens_row_class("DIFF", {"n_resolvable_params": 0}) == (
            "status-failed",
            "DIFF",
        )

    def test_non_scoring_outcomes_pass_through_unchanged(self):
        # BAD_TEST / UNSUPPORTED carry no n_resolvable_params and must render
        # exactly as classify_row says, badge = the raw outcome.
        assert gsm.sens_row_class("BAD_TEST", {}) == ("status-refused", "BAD_TEST")
        assert gsm.sens_row_class("UNSUPPORTED", {}) == ("status-refused", "UNSUPPORTED")

    def test_is_degenerate_pass_predicate(self):
        assert gsm.is_degenerate_pass("PASS", {"n_resolvable_params": 0}) is True
        assert gsm.is_degenerate_pass("PASS", {"n_resolvable_params": 1}) is False
        assert gsm.is_degenerate_pass("PASS", {"n_resolvable_params": None}) is False
        assert gsm.is_degenerate_pass("PASS", {}) is False
        assert gsm.is_degenerate_pass("DIFF", {"n_resolvable_params": 0}) is False


def _row(model_id, outcome, *, n_resolvable, max_abs_sx=0.0, state_span=0.0, comment=""):
    return {
        "model_id": model_id,
        "method": "sens/staggered",
        "outcome": outcome,
        "metric": "max_rel_err",
        "value": 0.0 if outcome in ("PASS", "DIFF") else None,
        "tol": 1e-4,
        "comment": comment,
        "timing": {
            "bngsim": {"integrate_warm_min_sec": 0.01},
            "amici": {"integrate_warm_min_sec": 0.02},
        },
        "extra": {
            "sens_method": "staggered",
            "n_params": 2,
            "n_param_candidates": 2,
            "state_passed": True,
            "n_resolvable_params": n_resolvable,
            "max_abs_sx": max_abs_sx,
            "state_span": state_span,
        },
    }


def _report():
    return {
        "_meta": {
            "suite": "amici_parity",
            "reference_engine": "amici",
            "regime": "forward_sensitivity",
            "versions": {"bngsim": "9.9.9", "amici": "0.0.0", "sundials": "7.0.0"},
            "tally": {"PASS": 2, "DIFF": 1},
            "n_jobs": 3,
            "n_models": 3,
            "sens_methods": ["staggered"],
            "param_cap": 0,
            "state_parity": {"n_state_diff": 0},
            "integration_tol": {"rtol": 1e-6, "atol": 1e-12},
            "concurrency": {"workers": 4},
            "hardware": {"cpu": "test", "physical_cores": 8},
        },
        "results": [
            # A genuinely vacuous pass: model does essentially nothing.
            _row(
                "vacuous_static",
                "PASS",
                n_resolvable=0,
                max_abs_sx=4.7e-18,
                state_span=2e-19,
                comment="3 sp x 2 par; state ok; DEGENERATE: no parameter column resolvable",
            ),
            # A healthy pass with real, resolvable sensitivities.
            _row("healthy", "PASS", n_resolvable=2, max_abs_sx=0.6, state_span=1.9),
            # A real disagreement — never NO SIGNAL.
            _row("diverged", "DIFF", n_resolvable=2, max_abs_sx=32.8, state_span=5.0),
        ],
    }


class TestSensMatrixRender:
    def test_vacuous_pass_renders_distinctly_from_a_real_pass(self, tmp_path):
        rpath = tmp_path / "report_sens.json"
        rpath.write_text(json.dumps(_report()))
        out = tmp_path / "sens.html"
        gsm.generate_html(rpath, out)
        html = out.read_text()

        # The distinct badge exists at all — the piece of #328 that was missing.
        assert "NO SIGNAL" in html

        # Exactly one row is a green PASS (the healthy one); the vacuous PASS is
        # a gray coverage gap, not green. `class='status-...'` matches only rows,
        # never the CSS rule (`.status-...`).
        assert html.count("class='status-passed'") == 1
        assert html.count("class='status-refused'") == 1  # the vacuous pass
        assert html.count("class='status-failed'") == 1  # the DIFF

        # The summary card discloses the vacuous count as a subset of the passes.
        assert (
            "<div class='card'><div class='k'>no signal</div><div class='v'>1</div></div>" in html
        )

        # The legend explains the badge and points at the disambiguating fields.
        assert "established nothing" in html
        assert "state_span" in html

    def test_a_report_with_no_vacuous_rows_shows_zero_and_no_badge(self, tmp_path):
        report = _report()
        # Make every row resolvable — nothing vacuous.
        report["results"] = [r for r in report["results"] if r["model_id"] != "vacuous_static"]
        report["_meta"]["tally"] = {"PASS": 1, "DIFF": 1}
        report["_meta"]["n_jobs"] = report["_meta"]["n_models"] = 2
        rpath = tmp_path / "report_sens.json"
        rpath.write_text(json.dumps(report))
        out = tmp_path / "sens.html"
        gsm.generate_html(rpath, out)
        html = out.read_text()
        assert (
            "<div class='card'><div class='k'>no signal</div><div class='v'>0</div></div>" in html
        )
        # No row carries the badge (the legend mentions it, so scope to the table).
        table = html.split('<div class="wrap">', 1)[1].split("</table>", 1)[0]
        assert "NO SIGNAL" not in table
        assert table.count("class='status-passed'") == 1
