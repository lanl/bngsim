#!/usr/bin/env python3
"""Test SBML feature extraction on a sample model."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Sibling module, importable only after the sys.path insert above.
from _sbml_features import extract_sbml_features  # noqa: E402

# Test on BIOMD0000000012 if it exists
test_model = HERE / "models" / "BIOMD0000000012" / "BIOMD0000000012_url.xml"

if test_model.exists():
    print(f"Testing on: {test_model}")
    features = extract_sbml_features(test_model)

    print("\nExtracted features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
else:
    print(f"Model not found: {test_model}")
    print("Run 'python materialize.py' first to place SBML files")
