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
2. **Check the conversion, which BioNetGen does not do.** Every conversion can be
   gated on a validation ladder, and a gate that catches drifted dynamics shows in
   the report (`report.ok` is False). The reverse direction (SBML → `.net`) is
   itself a key part of this: it lets the forward `.net` → SBML path be checked by
   round-trip identity. The ladder is a check, not a proof: its right-hand-side
   levels compare the model at sample points, so a conversion can pass and still
   differ. In issue #1499, `net_to_sbml`'s default `L1` gate passed an `if()` the
   writer had regrouped, whose SBML gives 20 where the `.net` gives 30.

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

# Check faithfulness with the BNG2.pl round-trip oracle (validate="bng2"): flatten
# the emitted .bngl via `BNG2.pl generate_network`, reload, and compare the ODE RHS,
# and each assignment-rule species' reported value, to the source. Needs BNG2.pl
# on $BNGPATH or PATH.
report = bngsim.convert.sbml_to_bngl("model.xml", "model.bngl", validate="bng2")
print(report.rhs_faithful)   # True when the round-trip reproduces the source
```

```bash
bngsim-sbml2bngl model.xml -o model.bngl            # SBML → cBNGL
bngsim-sbml2bngl model.xml -o model.bngl --gate     # + BNG2.pl round-trip check
```

cBNGL faithfulness is validated by the BNG2.pl round-trip rather than an in-tree
reader — a deliberate choice that keeps the check an *independent* oracle.

**Capability boundary (fail-loud).** Constructs a target cannot carry faithfully are
surfaced, not hidden: `tfun` table functions and time-varying compartment volumes
are **refused** under the default `strict=True` (`.net`→SBML); cBNGL additionally
refuses state-triggered events, live volumes, a model with no reactions (BNG2.pl's
`generate_network` aborts on one), and the BNG-dialect gaps it cannot express
(`!=`, non-finite parameters); SBML events are **dropped with a note**
(SBML→`.net`). Pass `--allow-lossy` (CLI) / `strict=False` (API) to downgrade a
refusal to a `ConversionWarning` and emit a best-effort artifact — the validation
ladder below still catches any numerical drift that results.

**A `.net` only bngsim reads as written.** An SBML species an assignment rule sets
is written `$`-fixed, beside the observable or function that holds its rule under
the species' own name. bngsim's reader takes the pair as the rule, so the file is
exact for bngsim, but BioNetGen's `run_network` holds the species at its initial
value for the whole run (`B := 2*A`: B = 20 at t = 5 there, 1.64 in bngsim). Its
rate laws read the rule through that observable or function, so only the
species' own column can be wrong there. The writer warns, with a `ConversionWarning`
naming those species, and never refuses, `strict=True` included;
`capability_report` and the reports `sbml_to_net` and `validate_conversion`
return list them under `bngsim_only`.

**A `.bngl` only bngsim reads back as written.** cBNGL carries such a species the
same way. A BNGL species name is a pattern (`@comp1::$B()`), which never equals
its rule's name, so the writer names the molecule after the rule instead:
`@comp1:$__ar_B()`, beside the function `B()`. bngsim's reader takes the pair as
the rule when it loads the `.net` that BNG2.pl writes from the file, so every
column comes back. BNG2.pl's own run still holds the species at its initial
value, so this too is a warning, listed under `bngsim_only`, and never a refusal.
`validate="bng2"` checks the column as well as the right-hand side, which cannot
show a held species (dy/dt = 0 on both sides), and reports the column's largest
difference as `max_ar_report_delta`.

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

Each run lands in SED-ML as BNG2.pl runs it. A run that starts from the model's
initial state, after any `setConcentration` / `addConcentration` / `setParameter`
actions, is a task on a derived model that makes those changes. A run that starts
where earlier runs left the model, such as the stimulation after an equilibration,
is a `repeatedTask` that resets the model once and then runs the earlier runs and
this one as ordered subtasks on one model state, with the actions between them as
`setValue` changes. In SED-ML, model state is not reset between subtasks unless
`resetModel="true"` asks for it. A bifurcate is its sweep up and back down, each
point continuing from the last. A parameter set to an expression of others,
`setParameter("k2","k1*3")`, is set again after each change to what it reads,
and at each point of a scan of it, since BNG2.pl keeps it as that expression. A
save made before `generate_network` is the network with the actions that made
what it saved. Every target names an id in the archive's SBML, under a declared
`sbml` prefix, and the lists are in the order the L1V3 schema gives. A simulator
that resets the model between subtasks does not follow that part of the spec,
and runs such a protocol differently: tellurium 2.2.13's own SED-ML runner, for
one.

Each experiment has a report: the time and the model's observables (its
species where it has none), read from the task that runs it. That task is the
experiment; any other task in the document is a part of one, such as the
earlier runs a chain replays. SED-ML reads a `repeatedTask`'s results as all of
its subtasks', so the report of a run that continues earlier ones holds their
rows first and its own last, and its name says so. A change to a parameter the
SBML computes by an `initialAssignment` (a derived parameter, or a species
whose seed amount names a parameter) is written after a `removeXML` of that
assignment, which would otherwise override it, as BNG2.pl's action replaces
the expression. The bngsim annotation holds the protocol as
parsed, with the `.bngl` named by its file name alone, not the path it was
read from.

**Starting from a `.bngl` model (the full export pipeline).** The converter's model
input is the flattened **`.net`**, so a rule-based model is first expanded to a
network with BioNetGen, then packaged — passing the same `.bngl` so its actions
become the SED-ML protocol *and* its source rides along. The `.net` has to be
the network as `generate_network` writes it, the state the protocol starts
from. `Model.from_bngl` writes that file with `net_out=`: it runs BioNetGen on
the model and the actions before `generate_network`, and stops there.

```bash
# 1. Flatten the rule-based model to its network, as generate_network writes it
#    (the .bngl needs a generate_network() action; rule-based models you
#    simulate already have one). bngsim says it leaves the later actions out:
#    they are the protocol, which step 2 reads from the .bngl.
python -c "import bngsim; bngsim.Model.from_bngl('model.bngl', net_out='model.net')"

# 2. Package a BioModels-ready, validated OMEX
bngsim-omex pack model.net --bngl model.bngl --gate full   # → model.omex
```

Running the whole file with `BNG2.pl model.bngl` does not give that network
for every protocol. At a `simulate` with no `suffix`, BNG2.pl writes
`model.net` again with the state of that moment, once a run or an action such
as `addConcentration` has moved it (generation, every `set*`, `add*` and
`reset*` action and every run's end mark it moved), and the protocol would then
be replayed on
top of a state it had already reached: `egfr_path.bngl`'s exported first run
started at Grb2 = 100163 for 100000 (issue #1683). The packer compares the
`.net`'s parameters and seed species with what the `.bngl` declares, and
refuses one that differs (`--allow-lossy` packs it with a warning). A `suffix`
on every `simulate` keeps BNG2.pl's `model.net` as generated too.

```python
# Equivalent API call. A validated, runnable deliverable: SBML + the real
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
dropped/lossy notes. So the validation verdict is auditable by anyone who
opens the archive, and a future reader can tell which bngsim version produced it.
`ok` and `lossy` judge the network. What the SED-ML's standard elements do not
hold of the protocol, such as a `reset_conc=>1` scan that BNG2.pl starts where
an earlier run left the model, is warned of at export and recorded as
`sedml_lossy`, in the report and in `bngsim-conversion.json`, and
`metadata.rdf` says how many such notes there are. A tool that runs the
standard elements runs something else there; bngsim reads the protocol from
its annotation.
The `created` timestamp is injectable (`created="…"` / `--created`) for byte-reproducible archives:
it is stamped both into the RDF and onto every zip entry, so two builds from the same
inputs and the same `created` are byte-identical however far apart they run. It applies
with `provenance=False` too, where the entry headers are the only place it lands.
Leave it out and the entries carry the build's wall clock instead, which is a fine
default but *not* reproducible.

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

The actions block is for the `.net` beside it, which holds what the protocol's
actions before `generate_network` did: it starts at that point, with any save
made before it made again, and names each species as the `.net` does
(`L_r` for the BNGL `L(r)`). BNG2.pl runs the pair with
`readFile({file=>"model.net"})` before the actions. A reset to a state saved
before generation whose values were the model's as declared cannot be made on
that `.net`, which holds the changed values: `omex_to_net` refuses it, and with
`strict=False` writes the block with a `# bngsim:` comment and a note in
`lossy`. An archive's
`load_full_protocol()` likewise names species and parameters by the master
SBML's ids, so it runs on `load_model()`; `load_full_protocol(entry)` names them
as another model entry does, the bundled `.net` by their BNGL names.

**Reading a SED-ML document another tool wrote.** `read_sedml_protocol` (and an
archive's `load_full_protocol`) reads a document without bngsim's annotation by
its tasks (issue #1662): each `task` is a run of its model from that model's
initial state, with the model's `changeAttribute` and `computeChange` changes
(a species' initial value, a parameter's value, a kinetic law's local parameter
as `_lp_<reaction>_<id>`, an addition of a term to the species as
`addConcentration`) as the steps before it; a `repeatedTask` that sweeps a
parameter is a `parameter_scan`, with a constant `setValue` beside the sweep as
a step before it, and one that does not sweep runs its subtasks in order on one
model state, resetting it first where `resetModel` says so. Tasks are
independent, so the state is reset between them, and a time course no task runs
is not run. A change the replay cannot apply (a compartment's size, an
`addXML`, a species' `initialAmount` without the SBML's compartment sizes, a
`setValue` that reads the range beside the swept parameter other than as an
expression of it, a sweep whose math reads the model's values, a task on a
second model file) is said with a `ConversionWarning` and recorded in the
spec's `lossy` list, which `run_protocol` refuses unless `allow_lossy=True`.
A sweep runs the value its `setValue`'s math gives at each iteration, over the
range it names: `k = 10^x` over x = 0, 1, 2 runs k = 1, 10, 100, and so does a
`functionalRange` or a second range of those values. A change's math is
evaluated once, where it applies, as SED-ML says; a `setValue` beside the sweep
that reads the range the swept parameter is set to is that parameter's
expression, and follows it through the scan. Pass `sbml=` to read a species
change written as an amount in its compartment. `read_sedml`, the single-run
reader, takes its first task's simulation, model and outputs the same way and
says which tasks it leaves unread (issue #1675). A data generator is an output
only where its math is one of its variables as it stands; one that computes a
value from them, such as `A + B`, is left out with a warning, since no output
selector names it.

```bash
bngsim-omex to-net model.omex -o model.net   # → model.net + model.bngl
```
