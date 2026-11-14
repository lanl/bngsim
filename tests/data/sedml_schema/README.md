# SED-ML Level 1 Version 3 schema

The XML Schema that `python/tests/test_sedml_schema.py` validates every SED-ML
document bngsim writes against (issue #1734). Both files are copies of the
upstream files, byte for byte, with their headers as written there.

| file | git blob | bytes |
|---|---|---|
| `sed-ml-L1-V3.xsd` | `80ebb8f56abdb8327cbcae40aefc33f866f77930` | 22,179 |
| `sedml-mathml.xsd` | `1698c19ccf7ab5da85a4ec569c9dbbaa965f9262` | 10,980 |

Source: <https://github.com/SED-ML/sed-ml>, `schema/level1/version3/`, at
commit `d129a9d74e2af84cdd96806bdec930a5cb99a6eb` (2017-10-05), the last to
touch `sed-ml-L1-V3.xsd`. `sed-ml-L1-V3.xsd` imports `sedml-mathml.xsd` by a
relative `schemaLocation`, so the two stay side by side, and validation needs
no network.

## Terms

- The SED-ML repository states no license, and `sed-ml-L1-V3.xsd` carries no
  license header. The schema is the normative part of the SED-ML L1V3
  specification, and other open-source tools ship the same file: Tellurium
  (Apache-2.0) and VCell (MIT).
- `sedml-mathml.xsd` is the SBML team's reduced MathML 2.0 schema. Its header
  reads "Copyright 2003-2006 California Institute of Technology, the Japan
  Science and Technology Corporation, and the University of Hertfordshire",
  licensed under the terms at <http://sbml.org/xml-schemas/LICENSE.txt>. The
  header is kept as written.

The files are test data only: no bngsim wheel or sdist ships them.

## Checking a document by hand

```bash
xmllint --noout --schema tests/data/sedml_schema/sed-ml-L1-V3.xsd simulation.sedml
```

`lxml.etree.XMLSchema`, which the test uses, runs the same libxml2 validator.
