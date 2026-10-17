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
- `rint(x)` — BNG's `rint()`: `floor(x + 0.5)`, so a half always rounds
  **up** (toward +∞), as BNG2.pl's `run_network` computes it. `rint(2.5)` is 3,
  `rint(-2.5)` is -2 and `rint(-0.5)` is 0. It is not the same as `round(x)`,
  which is C's `round` and rounds a half away from zero: `round(-2.5)` is -3.
  Below zero the two differ at every half. At and above zero they agree except
  where `x + 0.5` itself rounds, which `rint` keeps from BNG:
  `rint(0.49999999999999994)` is 1, `round` of it 0. (NFsim's function
  columns still use ExprTk's own `round`, which differs from C's only at such
  edges: `round(0.49999999999999994)` is 1 there.)

**Other math**: `abs`, `min`, `max`, `clamp`, `avg`, `sum`, `erf`, `erfc`
- `sign(x)` — returns -1, 0, or 1 (alias: `sgn`)

**Control flow**:
- `if(condition, true_value, false_value)` — ternary conditional. Condition is
  true when it is **nonzero** (`condition != 0`), in every bngsim backend: the
  ODE interpreter, compiled (`codegen=True`) models, SSA, PSA, NFsim and
  RuleMonkey. Example: `if(A_tot > 100, k_fast, k_slow)`

  A relational or logical condition (`>`, `<=`, `==`, `&&`, `||`, ...) always
  evaluates to exactly 0 or 1, so this is the same answer BNG2.pl gives. The
  rule matters only for a *bare* number used as the condition: BNG2.pl's
  `run_network` tests `condition > 0.5`, so a condition in `(0, 0.5]` or a
  negative one takes the true branch in bngsim and the false branch in
  `run_network`. For example, `if(c, 10, 20)` with `c = 0.3` gives 10 here and
  20 there. To get the same result in both, write the comparison you mean,
  e.g. `if(c > 0.5, 10, 20)` or `if(c != 0, 10, 20)`.

**Time**:
- `time()` — current simulation time (updated by CVODE/SSA at each step)

`time()` is the only way to read the clock. `t` is **not** an alias for it: it
is left free as an ordinary identifier, so that a model may name a parameter or
observable `t` (the BNGL counter idiom `Molecules t counter()` is why), matching
BNG2.pl. A bare `t` in an expression is that model symbol, and `t()` is the same
symbol written as a zero-argument call — the form BNG2.pl emits and which bngsim
accepts for any scalar. In a model that defines no `t`, both spellings fail to
compile rather than falling back to the clock.

A model may also declare a scalar named `time` itself: BNG2.pl rejects a
parameter named `time` but accepts an observable named `time`, and a
hand-written `.net` can declare either. The call form `time()` still reads the
clock (interpreted and compiled ODE, SSA, and the sensitivity layer); the bare
word `time` is the model's scalar. So
`k*time()*time` is the rate constant, times the clock, times the observable.
NFsim does not yet accept such a model: it refuses to load it with an error
naming `time` rather than guessing.

One place does treat `t` as the clock, and it is a different grammar: the
*index name* of a table function, where `time`, `T`, `Time()` and `t()` all
select simulation time. See [Table functions](../user-guide/table-functions.md).

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

Ported from BNG's `Util::Mratio` (W. S. Hlavacek, 2018). Evaluated by Gauss's
continued fraction with the modified Lentz method, and by an asymptotic expansion
for arguments the fraction cannot be trusted with. `a = 0` with `b > 0` and
`z > 0` is summed directly instead, as M(1, b+1, z), since M(0, b, z) = 1
(issue #768). Arguments none of these can vouch for are refused rather than
answered, because the fraction is silently wrong there. That includes every
`b <= 0` the asymptotic expansion cannot answer. See `expr_compat::mratio` in
`src/expression.cpp` for the region and the reasoning behind it.

`mratio` is differentiable in its third argument, so a rate law that reaches a
fitted rate constant through `z` (BNG writes `z = -1/Keq`) gets an analytic
forward-sensitivity gradient rather than CVODES' difference quotient. The first
two arguments have no closed-form derivative, so a model differentiating through
one of them falls back to the difference quotient.

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

A model may declare a parameter or observable named `and` or `or` (BNG2.pl
accepts both). `&&` and `||` are always the operators in such a model; the
conversion runs after model symbols are resolved, so it cannot pick up the
symbol. The bare words `and` and `or` then mean the model's symbol, so in that
model write the operators as `&&` and `||`, the form BNG2.pl emits.

## Case Sensitivity

BNGsim treats all identifiers as **case-sensitive**. Parameters `k3` and `K3`
are distinct variables with independent values. This matches BNG conventions
and prevents silent name collisions during expression evaluation.

## Adding Functions from Python

There is no Python hook for registering a callable as an expression function.
The evaluator is a C++ object, and the right-hand side runs with the GIL
released, so a Python callback would have to reacquire it on every evaluation —
serializing the thread-parallel batch sweeps that releasing it buys. Two
supported routes cover what such a callback would be used for.

**Declare the function in the model.** A `begin functions` block in the `.net`
file (or the equivalent in the BNGL, SBML or Antimony source) is where a
closed-form function belongs; it is compiled with the rest of the rate laws and
differentiated symbolically for sensitivities:

```
begin functions
    1 hill()  A_tot^2 / (Kh^2 + A_tot^2)
end functions
```

**Give a table function new data from Python.** A `tfun` the model declares
is an external signal supplied as data and interpolated at each step: a forcing
term, a dosing schedule, or a measured time course. After load it can take
other data, over the same index, without editing the model:

```python
model = bngsim.Model.from_net("model.net")  # declares signal() and drive() as tfuns
model.add_table_function("signal", file="signal_2021.tfun")
model.add_table_function("drive", times=[0, 1, 2, 5, 10], values=[0, 0, 1, 5, 5])
```

Nothing in a loaded model can call a table under a name its source does not
declare: `add_table_function` adds one, warns, and the table changes no run.

See the [table function guide](../user-guide/table-functions.md) for the `.tfun`
format, non-time index variables, and the interpolation and extrapolation rules.

## Full ExprTk Documentation

For the complete ExprTk expression syntax (operator precedence, string operations,
vector operations, etc.), see the upstream documentation:
https://github.com/ArashPartow/exprtk

BNGsim uses a subset of ExprTk focused on numerical expressions. String and
vector operations are disabled for compilation performance.
