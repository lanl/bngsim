#!/usr/bin/env python3
"""Check Jacobian density for all ODE models to find the right cutoff."""

import glob
import os

from bngsim._bngsim_core import NetworkModel

for net_file in sorted(glob.glob("bngsim/benchmarks/models/net/ode/*.net")):
    name = os.path.basename(net_file).replace(".net", "")
    try:
        model = NetworkModel.from_net(net_file)
        ns = model.n_species
        # Get sparsity info via C++ test — we'll compute density ourselves
        # by counting reactions touching each species pair
        nf = model.n_functions
        nr = model.n_reactions
        # Estimate density from reactions
        # For now just print species/reactions
        has_func = "FUNC" if nf > 0 else "ELEM"
        print(f"{name:<30} sp={ns:>5} rxn={nr:>6} {has_func}")
    except Exception:
        print(f"{name:<30} ERROR")
