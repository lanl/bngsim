# ExprTk

Vendored upstream `exprtk.hpp` used by `bngsim` expression evaluation.

## Source Of Truth

- Project homepage and official download: `https://www.partow.net/programming/exprtk/index.html`
- BNGsim's canonical vendoring source: `https://github.com/ArashPartow/exprtk.git`
- Canonical branch for refreshes: `master`
- Authoritative pinned provenance for the current vendored copy: `VENDOR.json`

BNGsim uses the author-maintained GitHub mirror as its vendoring input because
it provides immutable commits and a restartable checkout workflow. The Partow
site remains the upstream project homepage and official download location.

## Ownership Boundary

- `exprtk.hpp` should stay stock upstream unless a carry is explicitly
  documented in `VENDOR.json` and `bngsim/scripts/EXPRTK_VENDORING.md`.
- BNGsim-specific compatibility behavior belongs in the wrapper layer:
  `bngsim/include/bngsim/expression.hpp` and `bngsim/src/expression.cpp`
- Current local carries in `exprtk.hpp`: none

## Refresh Policy

- `bngsim/scripts/vendor_exprtk.py` is the only supported write path into
  `bngsim/third_party/exprtk`
- Builds must not download or rewrite ExprTk implicitly
- If this README and `VENDOR.json` disagree, trust `VENDOR.json`

For the supported preview, check, refresh, and validation commands, see
`bngsim/scripts/EXPRTK_VENDORING.md`.
