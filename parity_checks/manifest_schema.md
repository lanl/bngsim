# Unified corpus manifest schema

The contract for **corpus provenance manifests**: one *receipt per model* recording where it
came from, exactly which upstream version, the sha256 of the bytes we test against, whether we
patched it (and why), and its license. Machine-checkable companion:
[`manifest.schema.json`](manifest.schema.json) (JSON Schema draft 2020-12).

## Why one schema

Every corpus family (`bng_parity`, `rr_parity`, the BioModels stacks, DSMTS, …) records
provenance differently — or not at all. The bar is set by `bng_parity`: pinned upstream commit +
`sha256` per model + a repair ledger. This schema **generalizes that bar into a single record
shape** so all families express provenance the same way, and a fetch/verify script can validate
any corpus against one contract.

It is a **superset of bng_parity's existing record** (`source`, `origin{repo,commit,path}`,
`sha256`, `repairs`), so adopting it is additive: the only genuinely new required field is
`license`, plus the `origin.kind` discriminator to fit non-git upstreams (BioModels accessions)
alongside git ones.

A conforming manifest lets `fetch.py`/`fetch_sedml.py` re-fetch a corpus at its pinned version and
**assert every file's sha256 matches** — the actual reproducibility guarantee; the schema only
makes it checkable.

---

## Manifest file structure

A manifest is a JSON **object** with a header + a `records` array:

```json
{
  "schema_version": "1.0",
  "corpus": "rr_parity/temp-biomodels",
  "generated": "2026-07-03",
  "upstream_pin": {
    "source": "temp-biomodels",
    "repo": "sys-bio/temp-biomodels",
    "commit": "6a09daf46af1bb89e4857436b623a7b8720863ad"
  },
  "records": [ { …one record per model… } ]
}
```

- `schema_version` — this schema's version (`"1.0"`).
- `corpus` — the family / sub-corpus key (e.g. `bng_parity`, `rr_parity/temp-biomodels`).
- `generated` — ISO date the manifest was written (date only, so regeneration is reproducible).
- `upstream_pin` *(optional)* — corpus-level pin summary when every record shares one upstream
  version. Convenience; per-record `origin` is authoritative.
- `records` — the array of per-model receipts.

---

## The record

### Required fields (every family)

| Field      | Type    | Meaning |
|------------|---------|---------|
| `id`       | string  | Stable identifier, **unique within the manifest**. Accession corpora → the accession (`BIOMD0000000001`); file corpora → the relpath. |
| `source`   | string  | Upstream key (controlled vocabulary — see the source→license table). |
| `origin`   | object  | Where it came from + how it's pinned. Discriminated by `origin.kind`. |
| `sha256`   | string  | Lowercase hex sha256 of the **exact bytes we test against**. A re-fetch that doesn't hash-match is drift. |
| `license`  | object  | `{ spdx, notice }`. |

### Conditional / recommended

| Field      | Type    | Meaning |
|------------|---------|---------|
| `vendored` | string \| null | Repo-relative path **if the file is committed to git**; `null`/absent if gitignored and reconstructed by a fetch script. Distinguishes a committed corpus from a fetched one. |
| `patched`  | boolean | `true` iff `repairs` is non-empty. |
| `repairs`  | array   | Repair ledger; **absent/empty ⇒ pristine**. |

Family-specific extras are **permitted** as additional record keys (a validator ignores them):
`bng_parity` keeps `tier`/`methods`/`stochastic`/`has_free`/`companions`; the temp-biomodels
manifest keeps `model_id`/`artifact`/`horizon_source`.

### `origin` — three kinds

Discriminated by `origin.kind`; the kind determines which "version-or-hash" field is authoritative.

**`git`** — a versioned repo (bng_parity's sources, temp-biomodels):
```json
{ "kind": "git", "repo": "RuleWorld/RuleHub",
  "commit": "479d6d62a175572f28b2f6b6d7a376b6d1132da2",
  "path": "Examples/biology/aktsignaling/akt-signaling.bngl" }
```
Pin = `commit`.

**`rest-accession`** — a database with per-model versions but no repo commit (BioModels REST):
```json
{ "kind": "rest-accession", "service": "BioModels",
  "accession": "BIOMD0000000001", "version": "2",
  "url": "https://www.ebi.ac.uk/biomodels/BIOMD0000000001",
  "snapshot": "2026-07-03" }
```
Pin = `version` + `snapshot` (the date the REST snapshot was taken; BioModels has no global release
tag). `sha256` is what actually guarantees exactness.

**`url`** — a single immutable direct download (a commit-pinned raw URL, a release asset, a DOI):
```json
{ "kind": "url",
  "url": "https://github.com/…/raw/<commit>/workspace.mat",
  "retrieved": "2026-06-01" }
```

### `license`

```json
{ "spdx": "CC0-1.0", "notice": null }
```
- `spdx` — SPDX identifier, or `"unknown"` when the upstream states none (do **not** invent one).
- `notice` — repo-relative path to a LICENSE/NOTICE that **must travel with the copy** (MIT, CC-BY,
  LGPL), or `null` for public-domain grants (CC0) where no notice is legally required.

Source→license map:

| `source`         | Upstream                        | `spdx`          | Notice travels? |
|------------------|---------------------------------|-----------------|-----------------|
| `rulehub`        | RuleWorld/RuleHub               | `MIT`           | yes             |
| `rulemonkey`     | richardposner/RuleMonkey        | `MIT`           | yes             |
| `bngl_models`    | wshlavacek/BNGL-Models          | `CC-BY-4.0`     | yes (attribution)|
| `bngl_library`   | wshlavacek/BNGL-Models          | `CC-BY-4.0`     | yes (attribution)|
| `curated`        | wshlavacek/BNGL-Models          | `CC-BY-4.0`     | yes (attribution)|
| `biomodels`      | BioModels curated `BIOMD*`      | `CC0-1.0`       | no              |
| `biomodels`      | BioModels non-curated `MODEL*`  | `unknown`       | case-by-case    |
| `temp-biomodels` | sys-bio/temp-biomodels          | `CC0-1.0`       | no              |
| `dsmts`          | sbmlteam DSMTS                  | `LGPL-2.1-or-later` | yes             |
| `sbml-test-suite`| sbmlteam SBML Test Suite        | `LGPL-2.1-or-later` | yes             |
| `d2d`            | Data2Dynamics fit workspaces    | see upstream    | attribution by URL |
| `petab`          | PEtab benchmark collection      | `BSD-3-Clause`  | yes             |

### `repairs[]` — the patch ledger

Absent/empty ⇒ pristine (bytes == upstream at the pinned version). Each repair:

```json
{ "issue": "bngsim#173",
  "reason": "non-standard tolerance arg names abstol/reltol (BNG2.pl ignores them → default tol); canonicalized to atol/rtol so the intended setting is honored",
  "edits": 2,
  "patch_file": "corpus_repairs/ph_wave_equation.patch" }
```
- `issue` — tracking ref for the repair (optional).
- `reason` — **required** when a repair exists: why the upstream bytes were changed.
- `edits` — count of changed lines/tokens, for a count-guard that detects a re-vendor silently
  altering the file (optional but recommended).
- `patch_file` — repo-relative path to the pristine-vs-patched diff (or pristine copy) so the edit
  is **recoverable and reviewable**, or `null` if recoverability is provided out-of-band (e.g. a
  count-guarded repair block). A missing `patch_file` on a patched model is a *silent* patch — the
  failure mode this field exists to prevent.

---

## Worked examples (one per family)

### bng_parity — RuleHub (MIT, committed, pristine)
```json
{
  "id": "Examples/biology/aktsignaling/akt-signaling.bngl",
  "source": "rulehub",
  "origin": { "kind": "git", "repo": "RuleWorld/RuleHub",
    "commit": "479d6d62a175572f28b2f6b6d7a376b6d1132da2",
    "path": "Examples/biology/aktsignaling/akt-signaling.bngl" },
  "vendored": "fast/rulehub/Examples/biology/aktsignaling/akt-signaling.bngl",
  "sha256": "c89be2edd80d4406571b770c2f1ac39e37e5a2a84a843a5d06390e2ca0070f4f",
  "license": { "spdx": "MIT", "notice": "fast/rulehub/LICENSE" },
  "patched": false,
  "tier": "fast", "methods": ["ode"], "stochastic": false, "has_free": false, "companions": []
}
```

### rr_parity — temp-biomodels SED-ML (CC0, fetched/gitignored)
```json
{
  "id": "BIOMD0000000001/BIOMD0000000001_url.sedml",
  "source": "temp-biomodels",
  "origin": { "kind": "git", "repo": "sys-bio/temp-biomodels",
    "commit": "6a09daf46af1bb89e4857436b623a7b8720863ad",
    "path": "final/BIOMD0000000001/BIOMD0000000001_url.sedml" },
  "vendored": null,
  "sha256": "5705cb6916d38d26f153b06ec87948a1a89b29e70b5adff6c0fd63fc1dbd4a2d",
  "license": { "spdx": "CC0-1.0", "notice": null },
  "model_id": "BIOMD0000000001", "artifact": "sedml", "horizon_source": "template_sedml"
}
```

### biomodels — curated SBML via REST (CC0, fetched)
```json
{
  "id": "BIOMD0000000001",
  "source": "biomodels",
  "origin": { "kind": "rest-accession", "service": "BioModels",
    "accession": "BIOMD0000000001", "version": "2",
    "url": "https://www.ebi.ac.uk/biomodels/BIOMD0000000001", "snapshot": "2026-07-03" },
  "vendored": null,
  "sha256": "407fccc883f80e6d120e34d8b0810a206a5cc6535a26d4574611504e683beb08",
  "license": { "spdx": "CC0-1.0", "notice": null }
}
```

### harness — DSMTS (LGPL-2.1, committed, notice required)
```json
{
  "id": "00001/00001-sbml-l3v2.xml",
  "source": "dsmts",
  "origin": { "kind": "git", "repo": "sbmlteam/sbml-test-suite",
    "commit": "473e119dd57226c3a7a729d598f9007f06f781c3", "path": "cases/stochastic/00001/00001-sbml-l3v2.xml" },
  "vendored": "harness/sbml_test_suite/dsmts/cases/00001/00001-sbml-l3v2.xml",
  "sha256": "<sha256>",
  "license": { "spdx": "LGPL-2.1-or-later", "notice": null }
}
```

---

## Conformance checklist

A manifest conforms when:

1. It validates against [`manifest.schema.json`](manifest.schema.json).
2. Every `sha256` matches the bytes it points at (the vendored file, or a re-fetch at the pinned
   version). This is the actual reproducibility test — the schema only makes it *checkable*.
3. Every `origin` names a **pinned** version: `git`→`commit`, `rest-accession`→`version`+`snapshot`,
   `url`→an immutable URL. No `@main` / "latest" / bare branch. (The schema enforces this
   structurally.)
4. Every non-`unknown` `license.spdx` whose notice must travel has a non-null `notice` path
   pointing at a file that exists.
5. Every record with `patched: true` has ≥1 `repairs` entry with a `reason`; every `repairs` entry
   is recoverable via `patch_file` or a count-guarded repair block.
