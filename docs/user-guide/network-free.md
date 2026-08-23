# Network-free simulation (NFsim & RuleMonkey)

## Network-free simulation (NFsim and RuleMonkey)

For rule-based models with combinatorial complexity, use `Simulator` with a
network-free method token. BNGsim runs both vendored network-free backends
in-process; the `xml_path` argument points to the BNG-generated XML consumed by
NFsim or RuleMonkey.

```python
import bngsim

model = bngsim.Model.from_net("model.net")

# Rejection/null-event network-free simulation via NFsim:
sim = bngsim.Simulator(model, method="nf", xml_path="model.xml")
sim = bngsim.Simulator(model, method="nf_reject", xml_path="model.xml")
sim = bngsim.Simulator(model, method="nfsim", xml_path="model.xml")

# Exact non-local network-free simulation via RuleMonkey:
rm = bngsim.Simulator(model, method="rm", xml_path="model.xml")
rm = bngsim.Simulator(model, method="nf_exact", xml_path="model.xml")

result = rm.run(t_span=(0, 100), n_points=101, seed=42)
```

### NFsim connectivity option

BNGsim's in-process NFsim wrapper defaults to `connectivity=False`.

- `connectivity=False` uses the conservative full membership-update path.
- `connectivity=True` enables NFsim's inferred dependency-graph path.

Use `connectivity=True` only as an explicit opt-in:

```python
sim = bngsim.Simulator(
    model,
    method="nfsim",
    xml_path="model.xml",
    connectivity=True,
)
```

Current guidance:

- `connectivity=true` is correctness-clean on the supported NF benchmark suite.
- It is not a general timing win across that suite, so the wrapper default remains `False`.
- Prefer the default unless you have validated `connectivity=True` on your model and workload.

### Network-free method tokens

BNGsim uses algorithm-based method names (not tool brands) following the taxonomy
of Chylek et al. (2013) and Suderman et al. (2019):

| Token | Canonical | Algorithm | Status | Backend |
|-------|-----------|-----------|--------|---------|
| `"nf"` | `nf_reject` | Default network-free policy | ✅ Available | NFsim |
| `"nf_reject"` | `nf_reject` | Rejection/null-event (Yang et al.) | ✅ Available | NFsim |
| `"nf_exact"` | `nf_exact` | Exact non-local network-free | ✅ Available when built with RuleMonkey | RuleMonkey |

**Compatibility aliases** (accepted, normalized internally):

| Alias | Resolves to | Notes |
|-------|-------------|-------|
| `"nfsim"` | `nf_reject` | Legacy NFsim token |
| `"rulemonkey"` | `nf_exact` | Legacy RuleMonkey token |
| `"rm"` | `nf_exact` | Short RuleMonkey token |

Retired experimental network-free tokens such as `"nf_fixed"`, `"dynstoc"`, and
`"ds"` raise clear errors if requested. No silent fallback occurs.

You can inspect the normalization programmatically:

```python
from bngsim import normalize_method

canonical, dispatch = normalize_method("nf")
# canonical = "nf_reject", dispatch = "nfsim"

canonical, dispatch = normalize_method("nfsim")
# canonical = "nf_reject", dispatch = "nfsim"

canonical, dispatch = normalize_method("rm")
# canonical = "nf_exact", dispatch = "rulemonkey"
```

## How a rule's rate is counted

Both network-free engines have to answer the same question before they can fire
a rule: **how many distinct reactions does this rule represent right now?** Two
model features make that answer non-obvious, and getting either wrong changes a
trajectory by an integer factor rather than by a little. Both are settled the
same way in NFsim and RuleMonkey, so `method="nf"` and `method="rm"` agree.

### The symmetry factor

A reactant pattern with a non-trivial automorphism matches the same reaction
more than once. `A(b) + A(b) -> A(b!1).A(b!1)` matches each unordered pair of
molecules twice, once in each order, so a naive count doubles the propensity.

BNG2.pl computes the correction as `1/automorphisms/context-permutations` and
writes it into the XML as `symmetry_factor`, **independently of the rate law
type** — a symmetric rule carries `symmetry_factor="0.5"` with `type="Ele"`,
`type="Function"` and `type="MM"` alike. BNGsim applies it on all of them:
elementary rate constants, global functions, local functions, function products
and Michaelis-Menten (issue #195).

For Michaelis-Menten the factor goes on the **substrate count inside the law**,
not on the finished propensity (issue #282). The factor corrects a match
multiplicity, so the law is being handed `2N` where `N` exists, and
`σ·MM(2N, E)` equals `MM(σ·2N, E)` only where the law is linear. Above
saturation the two placements genuinely differ. Only the substrate needs it: an
MM rule does not transform its enzyme, and BNG counts reaction-centre
automorphisms, so an enzyme-side symmetry arrives as `symmetry_factor="1"`.

### Reactant patterns with no reaction centre

A reactant pattern that the rule never modifies is pure *context* — a
multi-subunit scaffold a rule reads but does not touch. Such a pattern is
counted **once per complex**, not once per way it embeds into that complex
(issues #281, #298, #300). Counting embeddings runs a multi-subunit catalyst two
or three times too fast.

### `TotalRate`

Under `TotalRate` the rate law states the whole propensity of the rule outright.
There is no match count to correct, so **the symmetry factor is not applied at
all** (issue #426). Applying it ran a `TotalRate` rule with a symmetric reaction
centre at half the rate the model asks for.

Two cautions on `TotalRate` generally. BioNetGen does not implement it for
network simulations — `RxnRule.pm` carries the note that it is implemented only
for XML network-free output — so `generate_network` writes the rate law into the
`.net` as an ordinary rate constant and the ODE integrates plain mass action.
A `TotalRate` model therefore has no BNG2 result to check a network-free run
against. It is also rejected by BNG2.pl on saturable, Michaelis-Menten, Hill and
Arrhenius laws, and on local functions.

### Stateful network-free sessions

For advanced workflows that mutate parameters or live particle counts between
segments, use the stateful session APIs:

```python
from bngsim import NfsimSession, RuleMonkeySession

with NfsimSession("model.xml") as nf:
    nf.set_param("kp1", 0.5)
    nf.initialize(seed=42)
    result = nf.simulate(0, 100, n_points=101)

with RuleMonkeySession("model.xml") as rm:
    rm.set_param("kp1", 0.5)
    rm.initialize(seed=42)
    result = rm.simulate(0, 100, n_points=101)
```

**PyBNF integration**: When `bngsim` is installed, `method=>"nf"` / `"nfsim"`
routes through the in-process NFsim backend, and `method=>"rm"` / `"rulemonkey"`
/ `"nf_exact"` routes through RuleMonkey when RuleMonkey support is compiled in.
