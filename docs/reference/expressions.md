# Expression language reference

BNGsim uses [ExprTk](https://github.com/ArashPartow/exprtk) as its expression
evaluation engine, replacing the older muParser used by BioNetGen's `run_network`.
ExprTk compiles expressions to bytecode for fast repeated evaluation during simulation.

## Built-in Constants

All constants use an underscore prefix to avoid collision with user parameter names.
They use SI units (matching BNG conventions).

| Constant | Value | Description |
|----------|-------|-------------|
| `_pi` | 3.14159265358979 | Pi (π) |
| `_e` | 2.71828182845905 | Euler's number |
| `_NA` | 6.02214076 × 10²³ | Avogadro's number (mol⁻¹) |
| `_kB` | 1.380649 × 10⁻²³ | Boltzmann constant (J/K) |
| `_R` | 8.314462618 | Gas constant (J/(mol·K)) |
| `_h` | 6.62607015 × 10⁻³⁴ | Planck constant (J·s) |
| `_F` | 96485.33212 | Faraday constant (C/mol) |

Example usage in a `.net` file function:
```
begin functions
    1 kT()  _kB * Temperature    # thermal energy
end functions
```

## Built-in Functions

BNGsim provides all standard ExprTk functions plus BNG-specific extensions:

**Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`,
`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

**Exponential/logarithmic**:
- `exp(x)` — e^x
- `log(x)` — **natural logarithm** (this matches BNG/C++ convention; NOT base-10)
- `ln(x)` — alias for `log(x)` (natural log)
- `log2(x)` — base-2 logarithm
- `log10(x)` — base-10 logarithm
- `sqrt(x)` — square root

**Rounding**: `floor`, `ceil`, `round`, `trunc`
- `rint(x)` — alias for `round(x)` (backward compatibility with BNG's `rint()`)

**Other math**: `abs`, `min`, `max`, `clamp`, `avg`, `sum`, `erf`, `erfc`
- `sign(x)` — returns -1, 0, or 1 (alias: `sgn`)

**Control flow**:
- `if(condition, true_value, false_value)` — ternary conditional. Condition is
  true when > 0.5. Example: `if(A_tot > 100, k_fast, k_slow)`

**Time**:
- `time()` — current simulation time (updated by CVODE/SSA at each step)
- `t()` — alias for `time()`

**Special functions**:
- `mratio(a, b, z)` — confluent hypergeometric ratio M(a+1,b+1,z)/M(a,b,z)

## The `mratio` Function

`mratio(a, b, z)` computes the ratio of confluent hypergeometric (Kummer)
functions:

```
mratio(a, b, z) = M(a+1, b+1, z) / M(a, b, z)
```

where M(a, b, z) = ₁F₁(a; b; z) is Kummer's confluent hypergeometric function.
This ratio arises in stochastic gene expression models (e.g., the steady-state
distribution of mRNA in a two-state promoter model).

Ported from BNG's `Util::Mratio` (W. S. Hlavacek, 2018). Uses series evaluation
with convergence check (tolerance 10⁻¹⁵, max 10,000 terms).

Example in a `.net` file:
```
begin functions
    1 mean_mRNA()  mratio(k_on/gamma, (k_on + k_off)/gamma, rho/gamma)
end functions
```

## Logical Operators

BNG2.pl emits C-style logical operators (`&&`, `||`) in function expressions.
BNGsim automatically converts these to ExprTk's keyword syntax:
- `&&` → `and`
- `||` → `or`

Example (these are equivalent):
```
if(A_tot >= 0 && A_tot <= 200, k_active, 0)    # BNG2.pl output
if(A_tot >= 0 and A_tot <= 200, k_active, 0)    # ExprTk native
```

## Case Sensitivity

BNGsim treats all identifiers as **case-sensitive**. Parameters `k3` and `K3`
are distinct variables with independent values. This matches BNG conventions
and prevents silent name collisions during expression evaluation.

## Registering Custom Functions (Python API)

The ExprTk evaluator supports user-registered functions from Python:

```python
model = bngsim.Model.from_net("model.net")

# Zero-arg function (e.g., external signal)
model.evaluator.define_function("signal", lambda: current_signal_value)

# One-arg function
model.evaluator.define_function("hill", lambda x: x**2 / (1 + x**2))
```

Custom functions can take 0–3 arguments. They are called during RHS evaluation,
so they should be fast (avoid I/O or heavy computation).

## Full ExprTk Documentation

For the complete ExprTk expression syntax (operator precedence, string operations,
vector operations, etc.), see the upstream documentation:
https://github.com/ArashPartow/exprtk

BNGsim uses a subset of ExprTk focused on numerical expressions. String and
vector operations are disabled for compilation performance.
