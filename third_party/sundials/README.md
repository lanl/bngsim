# SUNDIALS Fetch Metadata

BNGsim does not vendor SUNDIALS source into the repository.

Managed builds fetch the pinned official SUNDIALS release archive recorded in
`VENDOR.json`, or read a local copy of it named by `BNGSIM_SUNDIALS_ARCHIVE`
(`source_archive.cmake`), checked against the same sha256. Refresh that metadata only through
`bngsim/scripts/vendor_sundials.py`.

See `bngsim/scripts/SUNDIALS_VENDORING.md` for the supported workflow.
