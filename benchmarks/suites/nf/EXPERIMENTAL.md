These BNGL fixtures are external-NFsim canaries.

They are kept out of the default `run.py` correctness suite because
current standalone NFsim does not support them cleanly end-to-end yet.
They are opt-in canaries only — run with `BNGSIM_NF_INCLUDE_EXPERIMENTAL=1`.

- `t_dor2.bngl`: two-reactant DOR (DOR2) canary. Standalone NFsim currently
  rejects this case, so it should not be treated as a normal regression failure.
- `test_compartment_XML.bngl`: cBNGL / compartment canary.

All canary BNGL files live alongside the core models in
`../../models/bngl/nf/`; the runner selects them by name.
