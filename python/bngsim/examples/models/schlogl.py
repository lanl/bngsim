"""Schlogl's model: a bistable switch.

Steady states at X = 100 and X = 400 (stable) and X = 250 (unstable):
dX/dt = k1*X^2 - k2*X^3 + k3 - k4*X = -1e-4*(X - 100)*(X - 250)*(X - 400).
Exact steady states: bngsim.examples.steady_states("schlogl").
"""

import bngsim


def build() -> bngsim.Model:
    b = bngsim.ModelBuilder()
    b.add_parameter("k1", 0.075)
    b.add_parameter("k2", 1e-4)
    b.add_parameter("k3", 1000)
    b.add_parameter("k4", 16.5)
    b.add_species("X", 50)
    b.add_reaction(["X", "X"], ["X", "X", "X"], rate="k1")  # 2X -> 3X
    b.add_reaction(["X", "X", "X"], ["X", "X"], rate="k2")  # 3X -> 2X
    b.add_reaction([], ["X"], rate="k3")
    b.add_reaction(["X"], [], rate="k4")
    return b.build()
