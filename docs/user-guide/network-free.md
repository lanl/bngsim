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
