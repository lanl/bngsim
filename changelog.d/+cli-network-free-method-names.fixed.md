- **`bngsim run` and `bngsim check` refused `--seed` and `--xml` with
  `--method nf_reject`, `nf_exact` or `nf_fixed`.** The command knew only
  `nf`, `nfsim`, `rm` and `rulemonkey` as network-free, where `Simulator`
  takes every one of them. It now reads the same list `Simulator` resolves.
