"""0 -> A -> 0. Constant production and first-order decay.

Exact mean: bngsim.examples.exact("birth_death", t). Started at 0, the number
of A at time t is Poisson-distributed with that mean.
"""

import bngsim


def build() -> bngsim.Model:
    b = bngsim.ModelBuilder()
    b.add_parameter("ksyn", 5000)  # production rate
    b.add_parameter("kdeg", 0.5)  # decay rate constant
    b.add_species("A", 0)
    b.add_reaction([], ["A"], rate="ksyn")
    b.add_reaction(["A"], [], rate="kdeg")
    return b.build()
