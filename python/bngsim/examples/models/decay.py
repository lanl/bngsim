"""A -> 0. First-order decay.

Exact solution: bngsim.examples.exact("decay", t).
"""

import bngsim


def build() -> bngsim.Model:
    b = bngsim.ModelBuilder()
    b.add_parameter("k", 0.5)  # decay rate constant
    b.add_species("A", 100)
    b.add_reaction(["A"], [], rate="k")
    return b.build()
