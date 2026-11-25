- **`bngsim.run_bngl` ran a `parameter_scan` or `bifurcate` with
  `method=>"protocol"` and a `poplevel` as a PSA scan.** Such a scan runs the
  file's `begin protocol` block at each point, and BNG2.pl hands the block's
  lines none of the scan's options, so the `poplevel` reaches no run. The scan
  keeps its `protocol` method now, and is refused, as it is without a
  `poplevel`, as one a protocol replay does not run.
