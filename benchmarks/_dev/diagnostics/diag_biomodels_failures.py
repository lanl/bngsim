#!/usr/bin/env python3
"""Categorize BioModels sweep failures."""

import json
import re
from collections import Counter

with open("bngsim/benchmarks/suites/antimony/results/antimony_sweep_results.json") as f:
    results = json.load(f)

biomodels = [r for r in results if r["source"] == "biomodels"]

# G1 failures
g1_fails = [r for r in biomodels if not r["g1_load"]]
print(f"=== G1 LOAD FAILURES ({len(g1_fails)}) ===")
cats = Counter()
for r in g1_fails:
    err = r["error"]
    if "Undefined symbol" in err:
        m = re.search(r"Undefined symbol: '(\w+)'", err)
        sym = m.group(1) if m else "?"
        cats["Undefined symbol"] += 1
    elif "duplicate variable" in err.lower():
        cats["Duplicate variable"] += 1
    elif "register variable" in err:
        cats["Variable registration"] += 1
    else:
        cats["Other: " + err[:50]] += 1

for cat, count in cats.most_common():
    print(f"  {cat}: {count}")

# G3 failures
g3_fails = [r for r in biomodels if r.get("g2_ode") and not r.get("g3_xval")]
print(f"\n=== G3 CROSS-VALIDATION FAILURES ({len(g3_fails)}) ===")

# Categorize by error type
no_match = 0
high_err = 0
err_vals = []
for r in g3_fails:
    matched = r.get("matched_species", 0)
    err_val = r.get("max_rel_err")
    if matched == 0:
        no_match += 1
    else:
        high_err += 1
        if err_val:
            err_vals.append(err_val)

print(f"  No species matched: {no_match}")
print(f"  Species matched but err > 1e-3: {high_err}")
if err_vals:
    import numpy as np

    arr = np.array(err_vals)
    print(f"    Median err: {np.median(arr):.2e}")
    print(f"    < 1e-2: {sum(arr < 1e-2)}")
    print(f"    1e-2 to 1e-1: {sum((arr >= 1e-2) & (arr < 1e-1))}")
    print(f"    1e-1 to 1.0: {sum((arr >= 1e-1) & (arr < 1.0))}")
    print(f"    > 1.0: {sum(arr >= 1.0)}")

# Show some examples
print("\n=== Sample G3 failures ===")
for r in g3_fails[:10]:
    name = r["name"]
    matched = r.get("matched_species", 0)
    sp = r.get("n_species", 0)
    err = r.get("max_rel_err", "N/A")
    errstr = r.get("error", "")[:60]
    print(f"  {name}: sp={sp} matched={matched} err={err}")
