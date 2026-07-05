# Extending bngsim

## Adding a New Built-in Function (Developer Guide)

To add a permanent built-in function to BNGsim (like `mratio`), edit
`bngsim/src/expression.cpp`:

**Step 1: Implement the function logic.** Write a plain C++ function:
```cpp
static double my_func_impl(double x, double y) {
    return x * std::exp(-y);  // your math here
}
```

**Step 2: Create an ExprTk adapter class.** ExprTk requires inheriting from
`exprtk::ifunction<T>`. The constructor argument is the number of parameters:
```cpp
template <typename T>
struct MyFunction : public exprtk::ifunction<T> {
    MyFunction() : exprtk::ifunction<T>(2) {}  // 2 arguments
    T operator()(const T& x, const T& y) override {
        return static_cast<T>(my_func_impl(
            static_cast<double>(x), static_cast<double>(y)));
    }
};
```

**Step 3: Register in the `Impl` constructor.** In `ExprTkEvaluator::Impl::Impl()`:
```cpp
// Add as member:  MyFunction<double> my_func;
// Then in constructor:
symbol_table.add_function("myfunc", my_func);
```

**Step 4: Add to `reserved_names()`.** In the `reserved_names()` function at the
bottom of `expression.cpp`, add `"myfunc"` to the `names.functions` vector.

**Step 5: Add to `is_exprtk_reserved()` if the name is bngsim-specific.**
If `myfunc` is *not* on ExprTk's `reserved_symbols[]` list (check
`bngsim/third_party/exprtk/exprtk.hpp` lines 451-466) and is *not* on
BNG2.pl's `%functions` list (`bionetgen/bng2/Perl2/Expression.pm:56-96`),
add `"myfunc"` to the `is_exprtk_reserved()` set under the
"bngsim-registered function aliases" section. This activates the
mangling path so a user parameter with the same name registers under
the key `r_myfunc` instead of failing with a confusing "name already
registered" error. (Skipping this step is what caused the original
`sign`-collision bug — see the corresponding CHANGELOG entry under
[Unreleased] / Fixed.)

**Step 6: Test.** Create a `.net` test file that uses `myfunc(x, y)` in a function
expression, verify it compiles and evaluates correctly. If you added the
name to `is_exprtk_reserved()` in step 5, also add a regression test
with the name used as a user parameter — modeled on
`tests/data/sign_as_parameter.net` and
`python/tests/test_sign_as_parameter.py`.

Supported arities: 0 (use `allow_zero_parameters() = true`), 1, 2, or 3 arguments.
For 0-argument functions that read external state (like `time()`), store a pointer
in the adapter class. See `TimeFunction` and `MratioFunction` in `expression.cpp`
as examples.
