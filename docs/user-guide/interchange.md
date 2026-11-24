---
summary: "Convert between .net, SBML and compartmental BNGL, check each conversion, and read and write SED-ML and COMBINE (OMEX) archives."
---

# Format conversion and interchange (SBML, SED-ML, OMEX)

BNGsim converts BioNetGen networks to and from SBML (the model), SED-ML (the
simulation protocol) and OMEX (a COMBINE archive that bundles them). Use it to
publish a BioNetGen model, with its actions, in a form BioModels and journals
accept, and to check that the conversion kept the model's dynamics.

You don't need to convert to run SSA or PSA on an SBML model. They run
directly on `Model.from_sbml(...)`; see
[PSA on SBML / Antimony models](../about/benchmarks.md#psa-on-sbml--antimony-models).

## Convert a network

```python
import shutil

import bngsim

# The bundled binding model, A + B <-> C: the .net BNG2.pl generated from it.
shutil.copy(bngsim.examples.path("binding", "net"), "model.net")

report = bngsim.convert.net_to_sbml("model.net", "model.xml")
print(report.ok)         # True when every check passed
print(report.summary())  # counts and the result of each check

bngsim.convert.sbml_to_net("model.xml", "roundtrip.net")  # also bngsim.sbml_to_net
```

```bash
bngsim-net2sbml model.net -o model.xml
bngsim-sbml2net model.xml -o roundtrip.net
```

Each conversion returns a report. Check `report.ok`. The checks compare the
two models at sample points, so they can pass while the models still differ
somewhere else. Treat a pass as evidence, not proof.

What to watch out for:

- **`.net` to SBML is the more faithful direction.** SBML holds amounts,
  concentrations and compartment volumes; a plain `.net` can't.
- **Constructs a target can't hold are refused.** `net_to_sbml` refuses table
  functions (`tfun`) and compartment volumes that change in time. SBML events
  are dropped with a note in `sbml_to_net`. Pass `strict=False` (API) or
  `--allow-lossy` (CLI) to get a best-effort file with a `ConversionWarning`
  instead. The checks below still catch numerical drift.
- **Some species only BNGsim reads correctly.** An SBML species set by an
  assignment rule is written to `.net` as a `$`-fixed species, next to an
  observable or function with the rule under the species' name. BNGsim reads
  the pair as the rule. BioNetGen's `run_network` holds the species at its
  initial value instead (for `B := 2*A`, B is 20 at t = 5 there, 1.64 in
  BNGsim). Rate laws read the rule correctly in both, so only that species'
  own column differs. The writer warns with a `ConversionWarning` naming
  these species and never refuses them. `capability_report`, `sbml_to_net` and
  `validate_conversion` list them under `bngsim_only`.

## Compartmental BNGL (SBML to cBNGL)

A plain `.net` assumes unit volumes. To keep compartment volumes, convert to
compartmental BNGL, a `.bngl` with a `begin compartments` block. It carries
fixed compartment volumes, transport between compartments, and SBML events at
fixed times, which become a BNGL actions block.

```python
report = bngsim.convert.sbml_to_bngl("model.xml", "model_cbngl.bngl")
```

To check it with BNG2.pl, pass `validate="bng2"`. It generates the network
from the `.bngl`, reloads it, and compares the right-hand side and each
rule-set species to the source:

```python
report = bngsim.convert.sbml_to_bngl("model.xml", "model_cbngl.bngl", validate="bng2")
print(report.rhs_faithful)   # True when the round trip matches the source
```

```bash
bngsim-sbml2bngl model.xml -o model_cbngl.bngl
bngsim-sbml2bngl model.xml -o model_cbngl.bngl --gate     # with the BNG2.pl check
```

`validate="bng2"` needs BNG2.pl on `$BNGPATH` or `PATH`. It uses BNG2.pl so the
check is independent of BNGsim's own reader.

cBNGL refuses state-triggered events, volumes that change in time, a model
with no reactions (BNG2.pl's `generate_network` stops on one), and what BNGL
can't express, such as `!=` and non-finite parameters. `strict=False` turns
these into warnings, as above.

An event at a fixed time becomes a `simulate` up to that time, then a
`setConcentration` for each assignment. Its trigger time, delay and assigned
values are read at the start of the run, so a parameter that an
initialAssignment sets, or an assignment rule over constants, has the value
the run gives it. An event that reads a parameter a rate rule or another
event changes is refused, since its time or value isn't known before the run.

A trigger that is already true at the start of `t_span` fires there only if
its `initialValue` is false; otherwise the event never fires. Events at one
time apply in the order BNGsim executes them: a strict trigger such as
`time > 1` after the rest, then each in priority order, highest first, with no
priority counting as 0. Where that order doesn't settle what a species is
left at, the conversion is refused: two events of equal priority that set it
to different values, which BNGsim executes in a random order, or a delayed
event with a strict trigger, which rounding places before or after the others.

A species set by an assignment rule is handled as in `.net`. Its molecule is
named after the rule, `@comp1:$__ar_B()`, next to the function `B()`. BNGsim
reads the pair as the rule when it loads the `.net` BNG2.pl generates, but
BNG2.pl's own run holds the species fixed. This is a warning, listed under
`bngsim_only`. `validate="bng2"` checks the species' column as well as the
right-hand side, and reports the largest difference as `max_ar_report_delta`.

## Check a conversion

`net_to_sbml` and `sbml_to_net` take `validate=` set to `"L1"`, `"L2"`,
`"full"` or `None`. `net_to_sbml` defaults to `"L1"`, `sbml_to_net` to `"L2"`.
`"full"` runs every level:

| Level | Check | Fails the conversion? |
|-------|-------|--------|
| **L0** | the output passes its own format's validator | yes |
| **L1** | same species and reaction counts and topology | yes |
| **L2** | a round trip `X→Y→X` gives the same ODE right-hand side | yes |
| **L3** | trajectories agree | yes |
| **L4** | the right-hand sides are equal symbolically (sympy) | no |

```python
report = bngsim.convert.net_to_sbml("model.net", "model.xml", validate="full")
report.ok                      # False if any of L0-L3 failed
report.validation.level("L3")  # one level's result and metrics
```

```bash
bngsim-net2sbml model.net -o model.xml --gate full   # exits non-zero on failure
bngsim-validate-conversion model.net                 # check without keeping output
```

`bngsim-validate-conversion` works in either direction, chosen from the source
suffix, and prints a report for each level. L4 reports `equal`,
`inconclusive` or `not-equal`. It allows for round-off and never fails a
conversion the numeric checks pass.

`sbml_to_net`'s default L2 check catches an assignment rule or a time-dependent
input that a `.net` can't carry, rather than freezing it to a constant.

## Package an OMEX archive

SBML holds the model but not the protocol: which runs, over what time, with
what method. Give the packer the `.bngl` and it turns every `simulate` and
`parameter_scan`, with their parameter and concentration changes, into
SED-ML, and bundles it with the SBML in a COMBINE archive. The `.bngl` also
sets the time span of the L3 check.

Start from a rule-based model in two steps:

```bash
# 1. Generate the network as generate_network writes it. BNGsim skips the
#    later actions: they are the protocol, which step 2 reads from the .bngl.
python -c "import bngsim; bngsim.Model.from_bngl('model.bngl', net_out='model.net')"

# 2. Package a checked archive
bngsim-omex pack model.net --bngl model.bngl --gate full   # writes model.omex
```

The same from Python, with the bundled binding model's `.bngl` given one
`simulate` action, the protocol the archive carries:

```python
from pathlib import Path

bngl = bngsim.examples.path("binding", "bngl").read_text(encoding="utf-8")
Path("model.bngl").write_text(
    bngl + 'simulate({method=>"ode", t_end=>100, n_steps=>100})\n', encoding="utf-8"
)

report = bngsim.convert.net_to_omex(
    "model.net", "model.omex", bngl="model.bngl", gate="full",
)
```

```bash
bngsim-omex unpack model.omex      # look inside an archive
```

Watch out for:

- **Use the network as `generate_network` writes it.** Running the whole file
  with `BNG2.pl model.bngl` can overwrite `model.net`: at a `simulate` with no
  `suffix`, BNG2.pl rewrites it with the state at that moment, once a run or
  an action has changed it. The protocol would then replay on top of a state
  it already reached. The packer compares the `.net`'s parameters and initial
  species with the `.bngl` and refuses a mismatch (`--allow-lossy` packs it
  with a warning). A `suffix` on every `simulate` also keeps BNG2.pl's
  `model.net` as generated.
- **No `.bngl`, or no `simulate` in it,** gives a default protocol, a time
  course from 0 to 100, with a `ConversionWarning`. The SED-ML marks it as a
  default, so nobody mistakes it for the author's protocol.
- **A SED-ML file beside plain SBML:** `bngsim-net2sbml … --sidecar`.

### What the archive holds

- **The source.** By default (`include_source=True`; `--no-source` turns it
  off) the archive also holds the original `.net` and `.bngl`. The SBML stays
  the master entry, so SBML-only tools are unaffected.
  [BioModels accepts COMBINE archives with supporting files](https://www.ebi.ac.uk/biomodels/model/submission-guidelines-and-agreement).
- **How it was made.** By default (`provenance=True`; `--no-provenance` turns it
  off) the archive holds a COMBINE `metadata.rdf` (creator `bngsim <version>`,
  the date, a description) and `bngsim-conversion.json`, with the check level,
  each level's result, `ok`, `rhs_faithful`, `max_rhs_delta`, and any notes on
  what was dropped or approximated.
- **Protocol notes.** `ok` and `lossy` judge the network. Parts of the protocol
  that SED-ML's standard elements can't hold, such as a `reset_conc=>1` scan
  that BNG2.pl starts where an earlier run left the model, are warned about
  and recorded as `sedml_lossy`, in the report and in
  `bngsim-conversion.json`. `metadata.rdf` gives their count. Other tools run
  the standard elements; BNGsim reads the full protocol from its annotation,
  which names the `.bngl` by file name only.
- **Byte-identical builds.** Pass `created="…"` (`--created`) to fix the
  timestamp. It goes into the RDF and onto every zip entry, so two builds from
  the same inputs are byte-identical. It applies with `provenance=False` too.
  Without it, entries carry the build's clock time.

### How the protocol is written in SED-ML

Each run is written so that a simulator that follows the SED-ML spec runs it
as BNG2.pl does:

- A run from the initial state is a task on a model with that run's changes.
- A run that continues from earlier runs, such as a stimulation after an
  equilibration, is a `repeatedTask`. It resets the model once, then runs the
  earlier runs and this one as subtasks on one state, with the actions between
  them as `setValue` changes. Its report holds the earlier runs' rows first.
  SED-ML doesn't reset between subtasks unless `resetModel="true"`. A
  simulator that resets anyway, such as tellurium 2.2.13's SED-ML runner, runs
  it differently.
- A bifurcation is a sweep up and back down, each point starting from the
  last.
- Each experiment's report holds time and the observables (or the species, if
  there are no observables).
- Values that depend on others follow their inputs, as in BNG2.pl. A derived
  parameter (`setParameter("k2","k1*3")`) and a value an `initialAssignment`
  computes are set again after each change to what they read. A species whose
  initial amount names a parameter or an expression follows it only until a
  run has made its amount a number.
- A simulator computes an `initialAssignment` only when it loads the model, and
  the assignment would override a change to what it sets. So the export
  removes it (`removeXML`) before such a change.
- An assignment SED-ML can't restate, one that calls an SBML function, reads
  time, or uses `rem`, `quotient` or `implies`, keeps the value the simulator
  computed. So does an `addConcentration` to a species that is still an
  expression, when an action before `generate_network` changed its declared
  amount. Both are warned about and recorded
  in `sedml_lossy`. A `.bngl` that BNG2.pl accepts never has the first kind.

## Read an archive back (OMEX to `.net`)

`omex_to_net` takes a COMBINE archive, possibly with several SED-ML files,
and writes a `.net` plus a `.bngl` actions block with every experiment:

```python
report = bngsim.convert.omex_to_net("model.omex", "model.net")  # gate="full" by default
```

```bash
bngsim-omex to-net model.omex -o model.net   # writes model.net and model.bngl
```

The actions block goes with the `.net` beside it. That `.net` already holds
what the actions before `generate_network` did, so the block starts there,
repeats any save made before it, and names species as the `.net` does (`L_r`
for the BNGL `L(r)`). BNG2.pl runs the pair with `readFile({file=>"model.net"})`
before the actions. A reset to a saved state that held the model's declared
values can't be made on that `.net`, which holds the changed values.
`omex_to_net` refuses it; with `strict=False` it writes the block with a
`# bngsim:` comment and a note in `lossy`.

An archive's `load_full_protocol()` names species and parameters by the
master SBML's ids, so it runs on `load_model()`. `load_full_protocol(entry)`
names them as another entry does; for the bundled `.net`, by their BNGL names.

### SED-ML from other tools

`read_sedml_protocol` (and `load_full_protocol`) also read SED-ML another tool
wrote:

- Each `task` runs its model from the model's initial state, with the model's
  `changeAttribute` and `computeChange` changes applied first. A kinetic
  law's local parameter is named `_lp_<reaction>_<id>`. Tasks are
  independent. A time course no task runs isn't run.
- The tasks run in order, each once. A chain replays the tasks it names
  before its last, as `write_sedml_protocol` writes a run that continues
  earlier ones. Where the tasks before it left the model at the point the
  chain continues from, it continues; where they passed that point earlier,
  the state is saved there and reset to. A chain none of whose start was run
  runs in full from the initial state. A task or chain that is the start of
  one already run isn't run again, wherever the document lists it.
- A `repeatedTask` that sweeps a parameter is a `parameter_scan`. Each point
  runs the value the `setValue` math gives: `k = 10^x` over x = 0, 1, 2 runs
  k = 1, 10, 100. A `functionalRange` sweeps the same way. A constant
  `setValue` beside the sweep runs as a step before it. A `repeatedTask` that doesn't sweep runs its subtasks in
  order on one state, resetting where `resetModel` says.
- Pass `sbml=` to read a species change written as an amount in its
  compartment. Without it, that change is lossy.

What a replay can't apply, such as a compartment size change, an `addXML`, a
species' `initialAmount` without the SBML's compartment sizes, or a task on a
second model file, is warned about with a `ConversionWarning` and
recorded in the spec's `lossy` list. `run_protocol` refuses a lossy spec
unless you pass `allow_lossy=True`.

`read_sedml` reads one run: the first task's simulation, model and outputs,
and says which tasks it skipped. An output whose data generator computes a
value, such as `A + B`, is left out with a warning.
