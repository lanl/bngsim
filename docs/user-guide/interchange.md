# Format conversion & interchange (SBML · SED-ML · OMEX)

BNGsim converts BioNetGen networks to and from the community-standard interchange
formats — **SBML** (the BioModels model carrier), **SED-ML** (the simulation
protocol), and **OMEX/COMBINE** (the archive bundling them). Two things make this
worth reaching for:

1. **Publish a BioNetGen model to the standards world.** Take a `.net` (with its
   `.bngl` actions) and emit SBML, a SED-ML protocol, and a COMBINE/OMEX archive
   ready for [BioModels](https://www.ebi.ac.uk/biomodels/) deposit or a journal
   supplement. This goes **beyond BioNetGen's own SBML export**: BNGsim translates
   the model's *actions* (`simulate` / `parameter_scan`, with their overrides) into
   SED-ML and packages the result as OMEX — neither of which `writeSBML()` does.
2. **Prove the conversion is faithful — a guarantee BioNetGen does not provide.**
   Every conversion can be gated on a validation ladder and **fails loud** rather
   than silently emitting a model whose dynamics drifted. The reverse direction
   (SBML → `.net`) is itself a key part of this: it lets the forward `.net` → SBML
   path be checked by round-trip identity. The correctness check *is* the feature.

Five console scripts ship with the package, each backed by a Python API under
`bngsim.convert`.

> **You do not need to convert to run a stochastic method on an SBML model.**
> BNGsim's PSA (and SSA) run **directly** on `Model.from_sbml(...)` — see
> [PSA on SBML / Antimony models](../about/benchmarks.md#psa-on-sbml--antimony-models). Convert when you
> want a `.net`/`.bngl` artifact or a standards deliverable, not to unlock a method.

## Network conversion (`.net` ⇄ SBML)

```python
import bngsim

# .net → SBML (the more faithful direction: SBML carries amount/concentration
# semantics and non-unit compartment volumes the plain .net text cannot)
report = bngsim.convert.net_to_sbml("model.net", "model.xml")
print(report.ok)         # True when every check passed
print(report.summary())  # counts + per-level verdicts

# SBML → .net (reverse); also exposed at top level as bngsim.sbml_to_net.
# Defaults to validate="L2": reload the emitted .net and confirm it reproduces the
# source ODE right-hand side, so assignment-rule / time-dependent forcing the flat
# .net cannot carry is caught rather than silently frozen to a constant.
bngsim.convert.sbml_to_net("model.xml", "model.net")
```

```bash
bngsim-net2sbml model.net -o model.xml      # .net → SBML
bngsim-sbml2net model.xml -o model.net      # SBML → .net
```

## Compartmental BNGL (SBML → cBNGL)

Plain `.net` is flat and unit-volume, so it refuses models with non-unit or
cross-compartment volumes. Targeting **cBNGL** (`.bngl` with a `begin compartments`
block) recovers them: static compartment volumes, cross-compartment **transport**
(via a per-species signed-flux split), and **fixed-time SBML events** translated
into a BNGL actions block.

```python
# SBML → cBNGL (.bngl with a begin compartments block)
report = bngsim.convert.sbml_to_bngl("model.xml", "model.bngl")

# Prove faithfulness with the BNG2.pl round-trip oracle (validate="bng2"): flatten
# the emitted .bngl via `BNG2.pl generate_network`, reload, and compare the ODE RHS
# to the source. Needs BNG2.pl on $BNGPATH or PATH.
report = bngsim.convert.sbml_to_bngl("model.xml", "model.bngl", validate="bng2")
print(report.rhs_faithful)   # True when the round-trip reproduces the source RHS
```

```bash
bngsim-sbml2bngl model.xml -o model.bngl            # SBML → cBNGL
bngsim-sbml2bngl model.xml -o model.bngl --gate     # + BNG2.pl round-trip proof
```

cBNGL faithfulness is validated by the BNG2.pl round-trip rather than an in-tree
reader — a deliberate choice that keeps the check an *independent* oracle.

**Capability boundary (fail-loud).** Constructs a target cannot carry faithfully are
surfaced, not hidden: `tfun` table functions and time-varying compartment volumes
are **refused** under the default `strict=True` (`.net`→SBML); cBNGL additionally
refuses state-triggered events, live volumes, and the BNG-dialect gaps it cannot
express (`!=`, non-finite parameters); SBML events are **dropped with a note**
(SBML→`.net`). Pass `--allow-lossy` (CLI) / `strict=False` (API) to downgrade a
refusal to a `ConversionWarning` and emit a best-effort artifact — the validation
ladder below still catches any numerical drift that results.

## Validate the conversion (the L0–L4 ladder)

`net_to_sbml` / `sbml_to_net` take `validate={"L1", "L2", "full", None}`.
`net_to_sbml` defaults to `"L1"` (a fast structural + ODE-RHS round-trip);
`sbml_to_net` defaults to `"L2"` (the round-trip-identity RHS check). `"full"` runs
the complete ladder and gates on the hard levels:

| Level | Check | Gates? |
|-------|-------|--------|
| **L0** | syntactic validity — the target passes its own validator | yes |
| **L1** | structural equivalence — species/reaction counts + topology | yes |
| **L2** | round-trip identity — `X→Y→X` reproduces the ODE RHS | yes |
| **L3** | numerical equivalence — trajectories agree (scale-aware) | yes |
| **L4** | symbolic RHS equivalence (sympy) | no — best-effort |

```python
report = bngsim.convert.net_to_sbml("model.net", "model.xml", validate="full")
report.ok                      # False if any hard gate (L0–L3) failed
report.validation.level("L3")  # the per-level result + metrics
```

```bash
bngsim-net2sbml model.net -o model.xml --gate full   # exits non-zero on failure
bngsim-validate-conversion model.net                 # grade without keeping output
```

The standalone `bngsim-validate-conversion` grades either direction (inferred from
the source suffix) and prints a per-level report. L4 is **non-gating**: it reports
`equal` / `inconclusive` / `not-equal`, forgives floating-point round-off (and
symbolic cancellation) the engine cannot reduce, and never blocks a conversion the
numeric gates accept.

## Carry the simulation protocol — the OMEX deliverable

SBML carries structure + math only; the *simulation protocol* (which runs, over
what horizon, with what method) lived in the source `.bngl` and was discarded when
the `.net` was generated. Give the converter that `.bngl` and it recovers the
**whole actions block** — every `simulate` / `parameter_scan` with its parameter
and concentration overrides — into SED-ML, then bundles it with the SBML into a
COMBINE archive. The `.bngl` also drives the L3 gate over the model's *own*
horizon instead of a blanket one.

**Starting from a `.bngl` model (the full export pipeline).** The converter's model
input is the flattened **`.net`** (bngsim loads `.net`/SBML/Antimony, not rule-based
`.bngl`), so a rule-based model is first expanded to a network with BioNetGen, then
packaged — passing the same `.bngl` so its actions become the SED-ML protocol *and*
its source rides along:

```bash
# 1. Flatten the rule-based model to a network with BioNetGen (the .bngl needs a
#    generate_network() action — rule-based models you simulate already have one).
BNG2.pl model.bngl     # → model.net

# 2. Package a BioModels-ready, verified-faithful OMEX
bngsim-omex pack model.net --bngl model.bngl --gate full   # → model.omex
```

```python
# Equivalent API call. A verified-faithful, runnable deliverable: SBML + the real
# SED-ML protocol + manifest.xml, gated L0–L4 — plus the original .net and .bngl
# bundled for provenance (see below).
report = bngsim.convert.net_to_omex(
    "model.net", "model.omex", bngl="model.bngl", gate="full",
)
```

```bash
bngsim-omex unpack model.omex      # inspect the packaged archive's contents
```

**Provenance: the archive carries your rule-based source, not just the SBML.** By
default (`include_source=True` / drop `--no-source`) the archive also bundles the
original `.net` (a secondary model entry) and the rule-based `.bngl` (a `source`
entry) alongside the SBML. The SBML stays the `master` curated entry, so this is
non-breaking for SBML-only consumers — but a published deposit then carries the
modeller's *actual formulation*, not only its flattened SBML projection.
[BioModels accepts COMBINE archives with such supporting files](https://www.ebi.ac.uk/biomodels/model/submission-guidelines-and-agreement)
("model files along with the supporting documents … can also be submitted in
COMBINE archive format"); SBML receives full curation while the bundled BNGL/`.net`
ride along as the authoritative source. Pass `include_source=False` / `--no-source`
for a lean SBML + SED-ML archive.

**Provenance: the faithfulness verdict travels with the archive.** Also by default
(`provenance=True` / drop `--no-provenance`) the archive records *how it was made*:
a COMBINE-standard `metadata.rdf` (`dcterms` creator = `bngsim <version>`, the
creation date, a description — the channel BioModels/COMBINE tools read) and a
`bngsim-conversion.json` carrying the full **faithfulness verdict** — gate level and
per-level L0–L4 result, `ok` / `rhs_faithful` / `max_rhs_delta`, and any
dropped/lossy notes. So the *verified-faithful* claim is auditable by anyone who
opens the archive, and a future reader can tell which bngsim version produced it.
The `created` timestamp is injectable (`created="…"`) for byte-reproducible archives.

If you omit the `.bngl` (or it carries no `simulate` action), BNGsim still bundles
a **runnable default** protocol (a `t=0..100` uniform time course) — but it emits a
`ConversionWarning` and marks the SED-ML as a synthesized default, so a consumer
can never mistake the placeholder for the modeller's actual protocol. A SED-ML
sidecar can also be emitted next to a plain SBML conversion with
`bngsim-net2sbml … --sidecar`.

## Round-trip back from a published archive (OMEX → `.net`)

The reverse of packing: take a published COMBINE archive — model + SED-ML
protocol(s), possibly across several files — and recover an editable BioNetGen
workflow. `omex_to_net` writes the `.net` and composes every experiment from every
SED-ML entry into a single `.bngl` actions block, so the simulation protocol comes
back as runnable BNGL rather than being lost on import.

```python
# .omex → .net (+ a <stem>.bngl actions block carrying the composed protocol)
report = bngsim.convert.omex_to_net("model.omex", "model.net")  # gate="full" default
```

```bash
bngsim-omex to-net model.omex -o model.net   # → model.net + model.bngl
```
