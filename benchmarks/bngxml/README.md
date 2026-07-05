These BNGL fixtures are BioNetGen-to-XML emission tests, not standalone NFsim
runtime benchmarks.

Current BioNetGen emits inline `tfun(...)` data into XML, but current
standalone NFsim still requires file-backed TFUN attributes in the XML parser.

- `test_tfun_expr.bngl`: inline `tfun(...)` time-index XML fixture
- `test_tfun_observable.bngl`: inline `tfun(...)` observable-index XML fixture
