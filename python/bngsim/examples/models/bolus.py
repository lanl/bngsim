"""A -> 0, with a dose of 100 added at t = 5.

Exact solution: bngsim.examples.exact("bolus", t).
"""

import bngsim


def build() -> bngsim.Model:
    b = bngsim.ModelBuilder()
    b.add_parameter("k", 0.5)  # decay rate constant
    b.add_species("A", 100)
    b.add_reaction(["A"], [], rate="k")
    b.add_event("time() >= 5", {"A": "A + 100"}, name="dose")
    return b.build()
