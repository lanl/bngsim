# `_dev/` — development-only benchmark scripts

Scripts here are **not** part of paper-data generation and are **not** run by
the `run_all.py` orchestrator. They are development-time probes: dated
"Session NN" A/B optimization benchmarks (`bench_*.py`), quick sanity checks
(`test_*_quick.py`), corpus inventory (`triage_bngl.py`), and ad-hoc
debugging tools (`diagnostics/`).

Paper-data benchmark suites live in `../suites/`.
