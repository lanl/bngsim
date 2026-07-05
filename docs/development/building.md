# Building & contributing



## Build from source

```bash
git clone https://github.com/lanl/bngsim.git
cd bngsim

# Install in development mode.
# --no-build-isolation requires the build backend to be installed first:
#   pip install "scikit-build-core>=0.10" "pybind11>=2.13"
# Drop the flag (pip install -e .) to let pip provision it automatically.
pip install --no-build-isolation -e .

# Run tests
python -m pytest python/tests/ -q    # 309 Python tests
./build/tests/test_bngsim             # 32 C++ tests
```

## Project structure

```
bngsim/
├── CMakeLists.txt          # C++ build (SUNDIALS, ExprTk, libbngsim)
├── pyproject.toml          # Python package config (scikit-build-core)
├── README.md               # project overview
├── run_tests.sh            # test harness
├── include/bngsim/         # C++ public headers
│   ├── bngsim.hpp          # main include
│   ├── model.hpp           # NetworkModel
│   ├── simulator.hpp       # CvodeSimulator, SsaSimulator
│   ├── result.hpp          # Result struct
│   ├── expression.hpp      # ExprTk wrapper
│   └── types.hpp           # Species, Reaction, Observable types
├── src/                    # C++ implementation
│   ├── model.cpp
│   ├── cvode_simulator.cpp
│   ├── ssa_simulator.cpp
│   ├── net_file_loader.cpp
│   ├── expression.cpp
│   ├── result.cpp
│   ├── bngsim_api.cpp
│   └── _bngsim_core.cpp   # pybind11 bindings
├── python/bngsim/          # Python package
│   ├── __init__.py
│   ├── _model.py           # Model class
│   ├── _simulator.py       # Simulator class
│   ├── _result.py          # Result class
│   └── _exceptions.py      # Exception hierarchy
├── python/tests/           # Python test suite (143 tests)
├── tests/                  # C++ tests + test data
│   ├── data/               # .net files for testing
│   └── test_bngsim.cpp     # C++ unit tests
└── third_party/exprtk/     # vendored ExprTk header
```

## Running CI locally

```bash
# Install cibuildwheel
pip install cibuildwheel

# Build wheels for current platform
cibuildwheel bngsim --output-dir wheelhouse
```
