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
    b.add_reaction([], ["X"], law="k1*X^2")
    b.add_reaction(["X"], [], law="k2*X^3")
    b.add_reaction([], ["X"], law="k3")
    b.add_reaction(["X"], [], law="k4*X")
    return b.build()
