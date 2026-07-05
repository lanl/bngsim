"""
emit.py
-------
Render Supplementary Figures S1 (BioModels 4-panel scatter) and S2
(SBML events 4-panel scatter) as LaTeX fragments.

Reads ``results/figS1_biomodels.csv`` (non-events models) and
``results/figS2_events.csv`` (events-tagged models), plots a scatter
of each engine's wall-clock vs.\\ BNGsim, and emits two Markdown
report fragments:

* ``bngsim/benchmarks/reports/generated/biomodels_figS1.md``
* ``bngsim/benchmarks/reports/generated/biomodels_figS2.md``

When ``--engines bngsim,rr,amici`` has been run, the figure shows
the full 4-panel layout (ExprTk vs.\\ RR, ExprTk vs.\\ AMICI,
codegen vs.\\ RR, codegen vs.\\ AMICI).  In the current 2-engine
state (bngsim vs.\\ RR), only the ExprTk-vs-RR panel has data;
unrun panels render as empty axes with a "no data yet" placeholder
so the figure still compiles.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
BNGSIM_ROOT = BENCH_DIR.parents[2]  # repo root (the bngsim/ tree)
PAPER_FIG_DIR = BNGSIM_ROOT / "dev" / "paper"
sys.path.insert(0, str(BENCH_DIR.parents[1]))  # bngsim/benchmarks/

from _emit import write_generated  # noqa: E402

DEFAULT_FIGS1_CSV = BENCH_DIR / "results" / "figS1_biomodels.csv"
DEFAULT_FIGS2_CSV = BENCH_DIR / "results" / "figS2_events.csv"


def _load_pairs(csv_path: Path) -> list[dict]:
    """Return the rows of a biomodels figS*.csv, or [] if the file is missing."""
    if not csv_path.exists():
        return []
    with csv_path.open() as fh:
        return list(csv.DictReader(fh))


def _coerce(v) -> float | None:
    if v in (None, "", "None"):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def _plot(rows: list[dict], png_path: Path, *, title: str) -> dict:
    """Plot the 4-panel scatter; return a small summary dict for the caption."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("BNGsim vs.\\ libRoadRunner (ExprTk)", "bngsim_wall_s", "rr_wall_s"),
        ("BNGsim vs.\\ AMICI (ExprTk)", "bngsim_wall_s", "amici_wall_s"),
        ("BNGsim vs.\\ libRoadRunner (codegen)", "bngsim_wall_s", "rr_wall_s"),
        ("BNGsim vs.\\ AMICI (codegen)", "bngsim_wall_s", "amici_wall_s"),
    ]
    counts = {}

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.5))
    for ax, (label, xkey, ykey) in zip(axes.flat, panels, strict=False):
        pts = []
        for r in rows:
            # The current runner does not distinguish ExprTk vs codegen --
            # we use the same column for both pairs of panels.  When the
            # runner grows codegen_wall_s and amici_wall_s, the keys flip.
            xv = _coerce(r.get(xkey))
            yv = _coerce(r.get(ykey))
            if xv is not None and yv is not None:
                pts.append((xv * 1e3, yv * 1e3))  # plot in ms
        counts[label] = len(pts)

        if pts:
            ax.scatter(
                [p[0] for p in pts],
                [p[1] for p in pts],
                s=14,
                alpha=0.55,
                color="#1f77b4",
                edgecolors="none",
            )
            lo = min(min(p[0] for p in pts), min(p[1] for p in pts)) * 0.5
            hi = max(max(p[0] for p in pts), max(p[1] for p in pts)) * 2.0
            ax.plot([lo, hi], [lo, hi], "--", linewidth=0.8, color="black", alpha=0.4)
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.text(
                0.5,
                0.5,
                "no data yet",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#888",
                fontsize=11,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_title(label.replace(r"\\", "\\"), fontsize=10)
        ax.set_xlabel("BNGsim wall (ms)")
        ax.set_ylabel("other engine wall (ms)")
        ax.grid(True, which="both", alpha=0.2)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    fig.savefig(png_path.with_suffix(".pdf"))
    plt.close(fig)
    return {"counts": counts, "n_rows": len(rows)}


def _figure_tex(fig_name: str, label: str, caption: str) -> str:
    return "\n".join(
        [
            r"\begin{figure}[h]",
            r"\centering",
            f"\\includegraphics[width=\\textwidth]{{../{fig_name}}}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\end{figure}",
        ]
    )


def _caption(name: str, info: dict, kind: str) -> str:
    n_rows = info.get("n_rows", 0)
    if not n_rows:
        return f"{kind}: no data yet -- run the {name} suite to populate."
    return (
        f"BNGsim vs.\\ libRoadRunner and AMICI on {n_rows} cross-validated "
        f"BioModels {kind}.  Four panels: "
        r"(a) BNGsim ExprTk vs.\ libRoadRunner, "
        r"(b) BNGsim ExprTk vs.\ AMICI, "
        r"(c) BNGsim codegen vs.\ libRoadRunner, "
        r"(d) BNGsim codegen vs.\ AMICI.  "
        "Points above the diagonal indicate models where BNGsim is faster.  "
        "Panels with no data require a 4-engine biomodels run "
        "(``run.py --engines bngsim,rr,amici``) and a codegen pass."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figS1-csv", type=Path, default=DEFAULT_FIGS1_CSV)
    parser.add_argument("--figS2-csv", type=Path, default=DEFAULT_FIGS2_CSV)
    args = parser.parse_args()

    s1_rows = _load_pairs(args.figS1_csv)
    s2_rows = _load_pairs(args.figS2_csv)

    if not s1_rows:
        print(f"[emit] no Fig S1 data at {args.figS1_csv}; figure will be empty.")
    if not s2_rows:
        print(f"[emit] no Fig S2 data at {args.figS2_csv}; figure will be empty.")

    s1_png = PAPER_FIG_DIR / "fig_biomodels.png"
    s2_png = PAPER_FIG_DIR / "fig_biomodels_events.png"
    s1_info = _plot(s1_rows, s1_png, title="BioModels non-events")
    s2_info = _plot(s2_rows, s2_png, title="BioModels SBML events")

    tex_s1 = _figure_tex(
        "fig_biomodels.png", "fig:biomodels_4panel", _caption("biomodels", s1_info, "models")
    )
    tex_s2 = _figure_tex(
        "fig_biomodels_events.png",
        "fig:events_4panel",
        _caption("biomodels", s2_info, "models with discrete events"),
    )

    out1 = write_generated("biomodels_figS1", tex_s1)
    out2 = write_generated("biomodels_figS2", tex_s2)
    print(f"[emit] wrote {out1}")
    print(f"[emit] wrote {out2}")
    print(f"[emit] wrote {s1_png} (+ .pdf)")
    print(f"[emit] wrote {s2_png} (+ .pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
