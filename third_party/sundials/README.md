# SUNDIALS Fetch Metadata

BNGsim does not vendor SUNDIALS source into the repository.

Managed builds fetch the pinned official SUNDIALS release archive recorded in
`VENDOR.json`. Refresh that metadata only through
`bngsim/scripts/vendor_sundials.py`.

See `bngsim/scripts/SUNDIALS_VENDORING.md` for the supported workflow.
