# bngsim × official SBML Test Suite runner (GH #241)

This directory produces bngsim's **authoritative-equivalent SBML Test Suite
score** and the **committed unsupported-tags manifest** that makes that score
honest. It is the bridge between bngsim and the *official* SBML Test Suite
runner (`~/Code/sbml-test-suite/src/test-runner`), whose grading — not our
in-repo fair harness — is the community-standard yardstick.

Tracked, dev-only (the wheel packages only `python/bngsim`; nothing here bumps
the wheel). The single source of truth for the declared boundary is the shipped
module [`bngsim._sbml_unsupported`](../../../../python/bngsim/_sbml_unsupported.py).

Files:

| file | role |
|------|------|
| `bngsim_wrapper.py` | the wrapper the official runner invokes per case (`%d %n %o %l %v` → CSV) |
| `bngsim_wrapper.sh` | shell shim that runs the wrapper with the bngsim venv interpreter |
| `bngsim-unsupported-tags.txt` | **committed** manifest of declared-unsupported tags |
| `gen_manifest.py` | regenerates the manifest from the SSOT (`--check` to verify) |
| `score.py` | local reimplementation of the official grading → the score |
| `score_results.json` | per-machine scorer output (git-ignored) |

---

## §A — What "authoritative" means here

The official runner grades a tool in three steps: (1) invoke the tool's wrapper
to write a CSV per case; (2) compare that CSV to the reference within the
per-case tolerances; (3) classify each case into a `ResultType` using the tool's
declared **unsupported tags**. Its GUI is a Java/SWT/JNI application that is
impractical to build in this environment, so `score.py` **reimplements steps 2–3
faithfully** (ported line-for-line from `CompareResultSets.java`,
`WrapperConfig.getResultTypeInternal`, and `TestCase.matches`) and drives bngsim
through the **same** shared load+integrate+resolve code the committed wrapper
uses. The number it reports is the local equivalent of the official runner's.

To cross-check against the real GUI, register `bngsim_wrapper.sh` there (see §C)
and paste the manifest tags into the wrapper's "unsupported tags" field; the
per-category counts should match `score.py`.

---

## §B — The manifest (declared capability boundary)

bngsim integrates ODEs. It **refuses** — loud, fail-closed — three constructs it
cannot faithfully simulate, and does not attempt flux-balance test types:

| tag(s) | why unsupported |
|--------|-----------------|
| `CSymbolDelay` | `delay()` ⇒ delay-differential equation; no DDE solver |
| `AlgebraicRule` | non-empty algebraic rule ⇒ DAE constraint; no DAE solver |
| `FastReaction`, `MultipleFastReactions` | `fast="true"` ⇒ fast-equilibrium constraint; no such solver |
| `fbc` | flux-balance cases are `FluxBalanceSteadyState`, not `TimeCourse` |

Declaring a tag makes the official runner score a matching case `Unsupported`
("tool does not support the relevant tag(s)") instead of `Error`. RoadRunner
refuses the same DDE/DAE/fast models.

**Deliberately NOT declared** (documented honest `NoMatch`, not gamed):

- `CSymbolAvogadro` — bngsim uses the SI-exact constant `6.02214076e23`; three
  cases (`00960`, `00961`, `01323`) encode the older rounded value and are an
  honest numeric mismatch (GH #231).
- `RandomEventExecution` — supported reproducibly-per-seed (GH #242); against a
  single stochastic reference it may `NoMatch`, again an honest outcome.

Regenerate after changing the SSOT, and verify in CI/pre-commit:

```sh
python gen_manifest.py           # rewrite bngsim-unsupported-tags.txt
python gen_manifest.py --check   # exit 1 if stale
```

`python/tests/test_sbml_unsupported_manifest.py` pins the committed file to the
SSOT and pins the loader/simulator refusal messages to the SSOT labels.

---

## §C — Registering the wrapper with the official runner

In the runner (GUI *Preferences → wrappers*, or a headless wrapper config):

- **executable**: absolute path to `bngsim_wrapper.sh`
- **arguments**: `%d %n %o %l %v`
- **unsupported tags**: the comma-joined line from `bngsim-unsupported-tags.txt`
  (`AlgebraicRule, CSymbolDelay, FastReaction, MultipleFastReactions, fbc`)
- **output directory**: any writable dir

The shim runs `bngsim_wrapper.py` with `bngsim/.venv/bin/python`; override with
`BNGSIM_PYTHON=/path/to/python`. The wrapper is **fail-closed**: on any load
refusal, simulation failure, or internal error it writes no CSV (→ `Unsupported`
if the case matches a declared tag, else `Error`), and it never fabricates a
column (a variable it cannot report is omitted → `NoMatch` via
`requireAllColumns`).

---

## §D — Running the local scorer

```sh
# full suite (default suite dir = $SBML_TEST_SUITE_DIR or ~/Code/sbml-test-suite/cases/semantic)
python score.py

# reconcile against the fair harness (run.py) verdicts
python score.py --reconcile ../results/sbml_test_suite_results.json

python score.py --case 00042          # one case
python score.py --quick 100           # first 100
python score.py --effort low          # cheap strided subset
```

It prints per-category counts, the correct-among-attempted rate, the in-scope
TimeCourse rate (the fair-harness-comparable denominator), and every `Error`
case (the ones to investigate first — a failure with no declared excuse). Full
sweep is a few minutes; results go to `score_results.json`.

> **Editable-install caveat.** Python runs live from the tree, but the C
> extension must be fresh — confirm case `01000` (#248) passes before trusting a
> sweep (`python score.py --case 01000`). Rebuild with
> `uv pip install --no-deps -e .` if stale.

---

## §E — Current result (SBML Test Suite v3.3.0, 1823 cases)

| outcome | count | meaning |
|---------|------:|---------|
| **Match** | **1577** | correct within per-case tolerance |
| NoMatch | 3 | numeric mismatch — the Avogadro trio (§B), intentional |
| CannotSolve | 1 | `01244` — see below |
| Unsupported | 242 | declared-unsupported DAE/DDE/fast/fbc refusals |
| Error | 0 | failures with no declared excuse — **none** |

- **In-scope TimeCourse Match: 1577 / 1789 = 88.1%.**
- **Correct among attempted: 1577 / 1580 = 99.8%** (Match / (Match+NoMatch+Error)).
- Reconciliation vs the fair harness (`run.py`, which reported bngsim 1578):
  **1822/1823 agree**, one difference — `01244`.

**Why `01244` is `CannotSolve`, not `Match`.** `01244` is a no-op *empty*
`<algebraicRule/>` (states `0 = ∅`, no constraint), which bngsim loads and
simulates correctly — so the fair harness grades it `pass`. But because it
carries the `AlgebraicRule` component tag, the official runner **ignores** its
output and scores it `CannotSolve` once we declare `AlgebraicRule` unsupported.
The SBML Test Suite has no finer "trivial-AlgebraicRule" sub-tag, so declaring
the tag (which correctly excuses 124 real DAE cases) forfeits this one solvable
case. This is the honest, inherent cost of tag-granular declaration — not a bug,
and not something to game away by dropping the tag (which would turn 124 real
refusals into `Error`).

The authoritative bngsim number is therefore **1577 Match**; the fair harness's
1578 differs only by `01244`, fully accounted for above.
