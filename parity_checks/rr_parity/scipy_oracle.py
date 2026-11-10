"""An ODE oracle that shares no code with bngsim or SUNDIALS (issue #1559).

``overrides.NO_ORACLE_ADJUDICATED`` settles a row that RoadRunner refuses with an
oracle that does not share RoadRunner's failure mode. This is that oracle: the
model's right-hand side is built from the SBML text with libsbml, and scipy
integrates it. ``parity_checks/tests/test_no_oracle_adjudications.py`` re-runs it
against bngsim for every ``confirm`` entry, so the evidence an entry cites is a
test anyone can run, not a note.

It reads a deliberately narrow subset of SBML and raises ``Unsupported`` on
anything else, rather than integrate a model it has misread:

* compartments of constant size;
* species in concentration units (``hasOnlySubstanceUnits="false"``), with no
  conversion factor; boundary or constant species are held at their start value;
* reactions with a kinetic law, whose local parameters shadow globals, with
  constant stoichiometry;
* function definitions and initial assignments, which libsbml expands;
* assignment rules;
* arithmetic, powers, roots, ``exp``, ``ln``, ``log``, ``abs``, ``pi``, ``e``
  and ``time``.

No events, rate rules, algebraic rules, piecewise, relational or logical
operators, delays or other csymbols.

``coords="log"`` integrates ``u = log(x)``. The solution is then positive by
construction, so a model whose small species make a trial step go negative (and
a non-integer power of it NaN) needs no floor or clamp that would change the
model. It needs every start value positive. A species whose true solution
reached 0 would send ``u`` to minus infinity, and the integration fails rather
than return a wrong trajectory.
"""

from __future__ import annotations

import math

import libsbml
import numpy as np
from scipy.integrate import solve_ivp


class Unsupported(ValueError):
    """The SBML uses something this oracle does not read."""


_ARITH = {
    libsbml.AST_PLUS: "+",
    libsbml.AST_MINUS: "-",
    libsbml.AST_TIMES: "*",
    libsbml.AST_DIVIDE: "/",
}
_UNARY = {
    libsbml.AST_FUNCTION_EXP: "_exp",
    libsbml.AST_FUNCTION_LN: "_log",
    libsbml.AST_FUNCTION_ABS: "_abs",
}


def _abs(z):
    """|z| for a real z, and its analytic continuation for a complex one, so the
    complex-step derivative in ``self_rates`` sees through it."""
    return -z if np.real(z) < 0 else z


_NUMBER = (libsbml.AST_INTEGER, libsbml.AST_REAL, libsbml.AST_REAL_E, libsbml.AST_RATIONAL)


def _py(node, name_of, lit) -> str:
    """Python source for one libsbml math node. ``name_of`` resolves an identifier
    and ``lit`` a number, each to an expression that is an atom where it lands."""
    t = node.getType()
    kids = [node.getChild(i) for i in range(node.getNumChildren())]

    def py(k):
        return _py(k, name_of, lit)

    if t == libsbml.AST_INTEGER:
        return lit(node.getInteger())
    if t in _NUMBER:
        return lit(node.getReal())
    if t == libsbml.AST_NAME:
        return name_of(node.getName())
    if t == libsbml.AST_NAME_TIME:
        return "t"
    if t == libsbml.AST_CONSTANT_PI:
        return lit(math.pi)
    if t == libsbml.AST_CONSTANT_E:
        return lit(math.e)
    if t in _ARITH:
        args = [py(k) for k in kids]
        if t == libsbml.AST_MINUS and len(args) == 1:
            return f"(-{args[0]})"
        if t in (libsbml.AST_PLUS, libsbml.AST_TIMES) and args:
            return "(" + f" {_ARITH[t]} ".join(args) + ")"
        if len(args) == 2:
            return f"({args[0]} {_ARITH[t]} {args[1]})"
        raise Unsupported(f"{node.getName()} with {len(args)} arguments")
    if t in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER) and len(kids) == 2:
        return f"({py(kids[0])} ** {py(kids[1])})"
    # libsbml gives root and log their default degree (2) and base (10) as a first
    # child, so both always have two: (degree or base, argument).
    if t == libsbml.AST_FUNCTION_ROOT and len(kids) == 2:
        return f"({py(kids[1])} ** ({lit(1.0)} / {py(kids[0])}))"
    if t == libsbml.AST_FUNCTION_LOG and len(kids) == 2:
        return f"(_log({py(kids[1])}) / _log({py(kids[0])}))"
    if t in _UNARY and len(kids) == 1:
        return f"{_UNARY[t]}({py(kids[0])})"
    raise Unsupported(f"math node {node.getName() or t} with {len(kids)} arguments")


def _names_in(node) -> set[str]:
    out = {node.getName()} if node.getType() == libsbml.AST_NAME else set()
    for i in range(node.getNumChildren()):
        out |= _names_in(node.getChild(i))
    return out


class SbmlOde:
    """The ODE system an SBML model defines, as ``rhs(t, y)`` over its species.

    ``names`` are the state species in order, ``y0`` their start concentrations,
    and ``source`` the generated Python, for inspection.
    """

    def __init__(self, xml: str):
        doc = libsbml.readSBMLFromString(xml)
        if (
            doc.getModel() is None
            or doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR)
            or doc.getNumErrors(libsbml.LIBSBML_SEV_FATAL)
        ):
            raise Unsupported("libsbml could not read the model")
        if doc.getLevel() < 2:
            # Level 1 writes a stoichiometry as a numerator over a denominator.
            raise Unsupported("SBML Level 1")
        for option in ("expandFunctionDefinitions", "expandInitialAssignments"):
            props = libsbml.ConversionProperties()
            props.addOption(option, True)
            if doc.convert(props) != libsbml.LIBSBML_OPERATION_SUCCESS:
                raise Unsupported(f"libsbml could not {option}")
        m = doc.getModel()
        if m.getNumInitialAssignments():
            raise Unsupported("an initial assignment libsbml could not expand")
        if m.getNumEvents():
            raise Unsupported("events")
        if any(not r.isAssignment() for r in m.getListOfRules()):
            raise Unsupported("a rate or algebraic rule")
        if m.isSetConversionFactor():
            raise Unsupported("a conversion factor")

        size = {}
        for c in m.getListOfCompartments():
            if not c.getConstant() or not c.isSetSize():
                raise Unsupported(f"compartment {c.getId()} has no constant size")
            size[c.getId()] = c.getSize()
        value = dict(size)
        for p in m.getListOfParameters():
            if p.isSetValue():
                value[p.getId()] = p.getValue()
        rules = {r.getVariable(): r.getMath() for r in m.getListOfRules()}
        for v in rules:
            value.pop(v, None)

        names, y0, held, comp = [], [], {}, {}
        for s in m.getListOfSpecies():
            sid = s.getId()
            if s.getHasOnlySubstanceUnits():
                raise Unsupported(f"species {sid} is in amount units")
            if s.isSetConversionFactor():
                raise Unsupported(f"species {sid} has a conversion factor")
            comp[sid] = s.getCompartment()
            if sid in rules:  # its rule gives its value; any start value is unused
                continue
            if s.isSetInitialConcentration():
                c0 = s.getInitialConcentration()
            elif s.isSetInitialAmount():
                c0 = s.getInitialAmount() / size[s.getCompartment()]
            else:
                raise Unsupported(f"species {sid} has no start value")
            if s.getBoundaryCondition() or s.getConstant():
                held[sid] = c0
            else:
                names.append(sid)
                y0.append(c0)
        idx = {n: i for i, n in enumerate(names)}

        # Every number reaches the generated code as an element of the float64
        # array ``_k``: an atom, so -2 ** 2 cannot be read as -(2 ** 2), and a
        # numpy float, so a negative base to a fractional power is NaN, as it is
        # in RoadRunner, not a Python complex number.
        consts: list[float] = []

        def lit(x) -> str:
            x = float(x)
            if not math.isfinite(x):
                raise Unsupported(f"a value that is not finite ({x})")
            consts.append(x)
            return f"_k[{len(consts) - 1}]"

        def name_of(n: str) -> str:
            if n in idx:
                return f"y[{idx[n]}]"
            if n in rules:
                return f"a_{n}"
            if n in held:
                return lit(held[n])
            if n in value:
                return lit(value[n])
            raise Unsupported(f"symbol {n} has no value")

        lines = ["def rhs(t, y):"]
        pending = dict(rules)
        while pending:  # assignment rules, each after the rules it reads
            ready = [
                v for v, math_ in pending.items() if not (_names_in(math_) & (set(pending) - {v}))
            ]
            if not ready:
                raise Unsupported("assignment rules that read each other in a cycle")
            for v in ready:
                lines.append(f"    a_{v} = {_py(pending.pop(v), name_of, lit)}")
        terms: dict[str, list[str]] = {n: [] for n in names}
        level = doc.getLevel()
        for j, rx in enumerate(m.getListOfReactions()):
            if rx.isSetFast() and rx.getFast():
                raise Unsupported(f"fast reaction {rx.getId()}")
            if not rx.isSetKineticLaw():
                raise Unsupported(f"reaction {rx.getId()} has no kinetic law")
            law = rx.getKineticLaw()
            local = {}
            for p in law.getListOfLocalParameters() if level == 3 else law.getListOfParameters():
                if not p.isSetValue():
                    raise Unsupported(f"local parameter {p.getId()} has no value")
                local[p.getId()] = p.getValue()

            def name_in_law(n: str, local=local) -> str:
                return lit(local[n]) if n in local else name_of(n)

            lines.append(f"    r{j} = {_py(law.getMath(), name_in_law, lit)}")
            for refs, sign in ((rx.getListOfReactants(), -1.0), (rx.getListOfProducts(), 1.0)):
                for sr in refs:
                    if sr.isSetStoichiometryMath() or (level == 3 and sr.isSetId()):
                        raise Unsupported(f"variable stoichiometry in {rx.getId()}")
                    if level == 3 and not sr.isSetStoichiometry():
                        raise Unsupported(f"a stoichiometry {rx.getId()} does not set")
                    if sr.getSpecies() in idx:
                        terms[sr.getSpecies()].append(
                            f"{lit(sign * sr.getStoichiometry())} * r{j}"
                        )
        lines.append(f"    d = [0.0] * {len(names)}")
        for n in names:
            if terms[n]:
                lines.append(f"    d[{idx[n]}] = ({' + '.join(terms[n])}) / {lit(size[comp[n]])}")
        lines.append("    return d")

        self.names: list[str] = names
        self.y0 = np.array(y0, dtype=float)
        self.source = "\n".join(lines)
        namespace = {
            "_exp": np.exp,
            "_log": np.log,
            "_abs": _abs,
            "_k": np.array(consts, dtype=float),
        }
        exec(self.source, namespace)  # noqa: S102 -- source generated above from the SBML math
        self._rhs = namespace["rhs"]

    def rhs(self, t: float, y) -> np.ndarray:
        """d[species]/dt at ``(t, y)``, in the order of ``names``.

        ``y`` is taken as float64 first: Python's own ``**`` gives a complex number
        for a negative base and a fractional power, where numpy's gives NaN.
        """
        return np.asarray(self._rhs(t, np.asarray(y, dtype=float)), dtype=float)

    def self_rates(self, t: float, y, which=None) -> np.ndarray:
        """``df_i/dx_i`` at ``(t, y)`` for each species index in ``which`` (all by
        default), by complex step: ``Im f_i(y + i*h*e_i) / h`` with ``h = 1e-20 |y_i|``
        has no subtraction in it, so it is exact to rounding at any scale, where a
        finite difference small enough to stay near a species of 1e-30 is lost
        beside an O(1) rate.

        Not at a species at or near 0 under a fractional power: at 0 the step reads
        the analytic continuation of ``x**p``, not its derivative, and below about
        1e-280 ``h`` (floored at 1e-300 to stay a normal double) is no longer small
        beside the species. Callers ask about species above that. A rate whose
        ``f_i`` term is so small that ``df_i/dx_i * h`` underflows (``x**1.5`` at
        1e-250) reads 0: it moves its species by nothing over any horizon. Where the real
        ``f_i`` is not finite (a negative base to a fractional power, the log of a
        negative), the continuation is all the step could read, so it gives NaN.
        """
        y = np.asarray(y, dtype=float)
        idx = range(y.size) if which is None else list(which)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            real = self.rhs(t, y)
        out = np.empty(len(idx))
        for n, i in enumerate(idx):
            if not np.isfinite(real[i]):
                out[n] = np.nan
                continue
            h = max(1e-20 * abs(y[i]), 1e-300) if y[i] != 0 else 1e-200
            z = y.astype(complex)
            z[i] += 1j * h
            out[n] = np.imag(np.asarray(self._rhs(t, z))[i]) / h
        return out

    def solve(
        self,
        t_start: float,
        t_end: float,
        n_points: int,
        *,
        method: str,
        rtol: float,
        atol: float,
        coords: str = "linear",
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """``(time, values[n_time, n_species], names)`` on ``n_points`` over
        ``[t_start, t_end]``, in the shape ``rr_ode`` and ``bn_ode`` return.

        In log coordinates ``atol`` bounds the error in ``log(x)``, so it is a
        relative error in every species, however small. Raises ``RuntimeError``
        when scipy does not reach ``t_end`` with finite values.
        """
        grid = np.linspace(t_start, t_end, n_points)
        if coords == "log":
            if (self.y0 <= 0).any():
                raise Unsupported("log coordinates need every start value positive")

            def f(t, u):
                # A rejected trial step can overflow exp; the integrator discards
                # it, and a non-finite result at the end raises below.
                with np.errstate(over="ignore", invalid="ignore"):
                    x = np.exp(u)
                    return self.rhs(t, x) / x

            y0 = np.log(self.y0)
        elif coords == "linear":

            def f(t, y):
                # A trial step below 0 makes a fractional power NaN; the integrator
                # rejects it, and a non-finite result at the end raises below.
                with np.errstate(invalid="ignore"):
                    return self.rhs(t, y)

            y0 = self.y0
        else:
            raise ValueError(f"coords must be 'linear' or 'log', not {coords!r}")
        sol = solve_ivp(f, (t_start, t_end), y0, method=method, t_eval=grid, rtol=rtol, atol=atol)
        values = np.exp(sol.y.T) if coords == "log" else sol.y.T
        if (
            not sol.success
            or values.shape != (n_points, len(self.names))
            or not np.isfinite(values).all()
        ):
            raise RuntimeError(
                f"scipy {method} ({coords}) did not reach t = {t_end}: {sol.message}"
            )
        return grid, values, list(self.names)
