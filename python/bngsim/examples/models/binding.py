"""A + B <-> C. A and B bind to make the complex C.

Exact solution: bngsim.examples.exact("binding", t).
"""

import bngsim


def build() -> bngsim.Model:
    b = bngsim.ModelBuilder()
    b.add_parameter("kf", 0.01)  # binding rate constant
    b.add_parameter("kr", 0.1)  # unbinding rate constant
    b.add_species("A", 100)
    b.add_species("B", 80)
    b.add_species("C", 0)
    b.add_reaction(["A", "B"], ["C"], rate="kf")
    b.add_reaction(["C"], ["A", "B"], rate="kr")
    return b.build()
