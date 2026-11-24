- **`bngsim run` and `bngsim check` refused `--seed` and `--xml` with
  `--method nf_reject` or `nf_exact`.** The command knew only `nf`, `nfsim`,
  `rm` and `rulemonkey` as network-free, where `Simulator` takes those too. It
  now reads the list `Simulator` resolves, so `nf_fixed`, `dynstoc` and `ds`
  get `Simulator`'s own message that the method is unavailable here.
