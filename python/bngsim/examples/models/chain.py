"""A -> B -> C. A decay chain.

Exact solution: bngsim.examples.exact("chain", t).
"""

import bngsim


def build() -> bngsim.Model:
    b = bngsim.ModelBuilder()
    b.add_parameter("k1", 1.0)  # A -> B rate constant
    b.add_parameter("k2", 0.5)  # B -> C rate constant
    b.add_species("A", 100)
    b.add_species("B", 0)
    b.add_species("C", 0)
    b.add_reaction(["A"], ["B"], rate="k1")
    b.add_reaction(["B"], ["C"], rate="k2")
    return b.build()
