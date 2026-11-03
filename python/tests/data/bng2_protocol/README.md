# BNG2.pl's own output for three protocols

What BioNetGen 2.9.3 writes when it runs these files whole, as
`perl BNG2.pl <file>.bngl` in a scratch directory:

| file | source | what it holds |
|---|---|---|
| `egfr_path.gdat` | `benchmarks/models/bngl/ode/egfr_path.bngl` | the 120 s stimulation after a 1e5 s equilibration with `EGF` set to 0 |
| `nfkb.gdat` | `benchmarks/models/bngl/ode/nfkb.bngl` | the 1,200 s TNF dose after a 50,000 s rest, then the 5 h washout it continues into |
| `wofsy_goldstein_<ab>.scan` | `benchmarks/models/bngl/ode/wofsy_goldstein.bngl` | each of its nine `parameter_scan`s, one row per ligand concentration |

`python/tests/test_protocol_run_bng2.py` replays each file's actions with
`bngsim.run_protocol` on the network shipped as `benchmarks/models/net/ode/<name>.net`
(the seed state each file generates) and compares, so the comparison needs no
BioNetGen when it runs (issue #1366). To regenerate, run BNG2.pl over each source
and copy the files above.
