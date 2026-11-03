# Changelog fragments

One file per change. **Do not edit `CHANGELOG.md`** — a release commit writes it,
and nothing else does.

`CHANGELOG.md` has exactly one place a new entry can go: the top of
`## [Unreleased]` → `### Fixed`. Two branches open at the same time therefore edit
the same region of the same file, and git stops on a conflict — the default
outcome, not an unlucky one (17 of the 30 merges before issue #668 touched the
file). `.gitattributes` sets `CHANGELOG.md merge=union`, which fixes that for a
local `git merge` but not on github.com, whose merge machinery does not read
`.gitattributes`. A fragment has no shared anchor, so neither path has anything
to conflict over.

## Adding one

Create `changelog.d/<issue>.<category>.md`:

```
changelog.d/659.fixed.md
```

`<category>` is one of `added`, `changed`, `deprecated`, `removed`, `fixed`,
`security` — Keep a Changelog's vocabulary, lowercased. Rarer forms:

| file | when |
|---|---|
| `659.fixed.md` | the normal case |
| `659.fixed.2.md` | a second entry of the same kind under one issue |
| `659.added.md` + `659.fixed.md` | one change that is both |
| `+short-slug.fixed.md` | a change with no issue number |

The file holds the bullet **exactly as it will read in `CHANGELOG.md`**, leading
`- ` included. Assembly is concatenation, so nothing reflows your text and a
nested list or a code span survives byte-for-byte:

```markdown
- **One sentence that says what was wrong and for whom (issue #659).** Then two
  or three more that say what the fix changes, and what a reader should do
  differently. Continuation lines are indented by two spaces; a nested list is
  `  * `.
```

## Keep it short

**1,200 characters, hard limit.** Aim for well under it — 400 to 600, a short
paragraph.

That number is arithmetic, not taste. GitHub rejects a release body over 125,000
characters, and `.github/workflows/release.yml` publishes the version's
`CHANGELOG.md` section as exactly that body. The largest release meant to fit
carried 64 entries; at 1,200 each that is 76,800, comfortably inside. At the rate
entries were actually being written — 2,650 characters each — that same release
came to 169,635, and `ci/release_notes.py` truncated it with a footer pointing
back at the file. 0.13.0 and 0.12.0 both shipped incomplete release notes that
way. 0.17.0 carried 468 entries from six weeks of work, more than any workable
per-entry limit can fit, and was published truncated on purpose.

An entry says **what changed and what it affects**. The evidence, the
measurements and the reasoning belong in the issue and the commit message, which
is where a reader who wants them will look, and which no character limit
constrains.

## Checking your work

```bash
python3 ci/changelog.py check     # names, shape, size — also a pre-commit hook
python3 ci/changelog.py render    # what the assembled section will look like
```

CI additionally asks that any branch changing `python/bngsim/`, `src/` or
`include/` stages a fragment. A change genuinely invisible to users — a refactor,
a test, a CI fix — is exempt by labelling the pull request **`changelog exempt`**.

## What happens at a release

```bash
python3 ci/changelog.py build --version <x.y.z>
```

folds every fragment into a new `## [<x.y.z>] - <date>` section, in Keep a
Changelog order with the highest issue number first, deletes the fragment files,
and refuses to proceed if the result would publish truncated, unless
`--allow-truncation` is passed. Commit that with
the version bump; `ci/release_notes.py` reads the assembled section at tag time,
unchanged.
