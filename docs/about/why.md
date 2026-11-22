---
summary: "Why a simulator is needed on top of an ODE solver such as CVODES, why a correct simulator is hard to build, and what BNGsim offers beyond SciPy and NumPy, with references."
---

# Why BNGsim?

Three questions arise about BNGsim and tools like it, such as libRoadRunner
(Somogyi et al., 2015; Welsh et al., 2023), AMICI (Fröhlich et al., 2021), and
COPASI (Hoops et al., 2006). Why not pass a model directly to an ODE solver such
as CVODES? Why is it hard to build a simulator that gives correct results? What
does BNGsim offer that SciPy and NumPy do not? We answer these questions here.

## Why not call CVODES directly?

CVODES, part of SUNDIALS (Hindmarsh et al., 2005; Gardner et al., 2022), solves
initial value problems of the form dy/dt = f(t, y, p). It is BNGsim's default
integrator; BNGsim has no ODE integration method of its own. CVODES provides
stiff and nonstiff methods, dense and sparse linear solvers, forward and
adjoint sensitivity analysis (Serban and Hindmarsh, 2005), and location of roots
of user-supplied functions. It requires code that evaluates f and, ideally, the
Jacobian ∂f/∂y and the derivatives ∂f/∂p used in sensitivity analysis. A model
file provides none of these. Most of what a simulator does is bridge this gap.

**A model is not a system of ODEs.** An SBML model consists of reactions with
rate laws, rules that set a variable by a formula or by an ODE, initial
assignments, function definitions, compartments that may change size, and
events with triggers, delays, and priorities (Hucka et al., 2003, 2019; Keating
et al., 2020). The ODEs and discrete state changes these elements imply must be
derived exactly as the specification defines them. For example, when a
compartment grows, the concentration of a species in it falls even if no
reaction consumes the species. A BNGL model is a set of rules, each of which
defines a class of reactions (Faeder et al., 2009; Harris et al., 2016). The
rules must be expanded into the reaction network they imply, and this network
can be large, because molecules with multiple sites can be modified and
combined in many ways (Hlavacek et al., 2003, 2006).

**Derivatives.** Most ODE models of biological systems are stiff (Städter et
al., 2021), so an implicit method is needed, and it needs the Jacobian or an
approximation of it. Gradient-based parameter estimation using sensitivity
equations outperformed alternative methods in a comparison by Raue et al.
(2013). Where it can, BNGsim derives the Jacobian and ∂f/∂p symbolically, using
SymPy (Meurer et al., 2017); where it cannot, it falls back to finite
differences and reports that it did. By default, BNGsim compiles generated C
code for large models and for all sensitivity runs; small models are otherwise
evaluated by an interpreter. Other
tools take similar approaches: libRoadRunner compiles SBML to machine code using
LLVM (Somogyi et al., 2015), and AMICI generates C++ code (Fröhlich et al.,
2021). In this sense, a simulator is a compiler.

**Events.** CVODES finds a root of a function g(t, y) by detecting a change of
sign over an integration step; a root without a change of sign will probably be
missed ([CVODES documentation](https://sundials.readthedocs.io/en/latest/cvodes/Mathematics_link.html)).
When it finds a root, CVODES returns control to the caller. The simulator must
then apply the event's state changes, order events that occur at the same time,
decide whether a trigger that is true at the start of a simulation fires, and
restart integration. Sensitivities must also be updated, because they jump at
an event whose timing depends on a parameter (Galán et al., 1999; Hiskens and
Pai, 2000; Fröhlich et al., 2017a). A rate law with a condition, such as
`if(t >= tau, k, 0)`, requires the same treatment.

**Other methods.** Some methods are not ODE integration at all: exact
stochastic simulation (Gillespie, 1976, 1977), the partial-scaling
approximation (Lin et al., 2019), network-free simulation (described below),
and calculation of steady states. Each of these methods needs the same model,
interpreted the same way.

An analogy familiar to physicists is circuit simulation. SPICE (Nagel, 1975)
turns a netlist and device models into a stiff system of differential-algebraic
equations and integrates it. Much of the value of a circuit simulator lies in
this translation, and the same is true of a simulator for biochemical models.

## Why is it hard to build a correct simulator?

**Ill-posed problems.** BNGsim is meant to accept any model that it is given and
either simulate it or refuse it. Its inputs include models from repositories
such as BioModels (Malik-Sheriff et al., 2020) and models evaluated at parameter
values chosen by an optimizer. We cannot assume that the resulting initial value
problem is well posed. For continuous f, a solution exists, but only locally,
and it is guaranteed to be unique if f is Lipschitz continuous in y (Hairer et
al., 1993). If the reaction 2A → 3A has rate kA², then dA/dt = kA², and the
solution becomes infinite at t = 1/(kA₀). The rate law k√A is not Lipschitz
continuous at A = 0, and dA/dt = k√A with A(0) = 0 has infinitely many
solutions, among them A = 0 and A = k²t²/4. A rate law with a condition on the
state, such as `if(X < 1, k1, k2)`, makes f discontinuous. A trajectory can then
slide along the switching surface, and the appropriate solution concept is that
of Filippov (Filippov, 1988; Cortés, 2008). A simulator should detect such cases
and report them, rather than return a trajectory that is not a solution.
BNGsim does not detect all of them; known gaps are tracked as
[correctness concerns](https://github.com/lanl/bngsim/labels/correctness%20concern)
in our issue tracker.

**Discontinuities.** A model with events is a hybrid dynamical system (Goebel
et al., 2009). Locating events reliably is a problem in its own right (Shampine
and Thompson, 2000). Several events can occur at the same instant, and an event
can coincide with a switch in a rate law. Sensitivities can jump at a
discontinuity, by an amount given by the saltation matrix (Hiskens and Pai,
2000; Leine and Nijmeijer, 2004). When an event and a switch coincide, the
derivative may not exist. BNGsim checks whether the two commute and refuses the
run if they do not. A finite-difference check fails here: a central difference
centered on a kink returns the average of the two one-sided slopes.

**Stiffness, scale, and positivity.** Concentrations and time scales in one
model can span many orders of magnitude, so implicit methods are needed
(Shampine and Gear, 1979; Hairer and Wanner, 1996), and the choice of solver
settings strongly affects results (Städter et al., 2021). Within error
tolerances, a concentration can become slightly negative, and a rate law that
takes its square root then fails. Preserving non-negativity is a problem of its
own (Shampine et al., 2005). Conservation laws make the Jacobian singular, so a
steady-state solver must first reduce the system (Vallabhajosyula et al., 2006).

**Floating-point arithmetic and compilers.** A simulator that generates and
compiles code inherits the behavior of the compiler on the user's machine. A
compiler may fuse a multiplication and an addition into one operation with one
rounding, depending on its default target architecture (Monniaux, 2008). We
found that BNGsim, built with the default compilers of two releases of Ubuntu,
rounded differently from the first integration step, and that results near
discontinuities changed as a result. BNGsim now disables this contraction, both
in its own code and in the code it generates. Compilers also have bugs (Yang et
al., 2011). The C front end of the MIR just-in-time compiler, which BNGsim can
use in place of a system compiler, miscompiled one of BNGsim's generated
routines. We rewrote the routine to avoid the problem.

**No reference answer.** For most inputs, the correct output of a scientific
program is unknown (Kanewala and Bieman, 2014; Barr et al., 2015). Few models
have analytic solutions. What remains are test suites with reference results,
such as the [SBML Test Suite](https://github.com/sbmlteam/sbml-test-suite) and,
for stochastic simulators, the Discrete Stochastic Models Test Suite (Evans et
al., 2008), and comparison with independent implementations, known as
differential testing (McKeeman, 1998). Agreement between independent codes is
evidence, not proof. Hatton (1997) found that independent implementations of the
same seismic data processing algorithms, applied to the same data, disagreed. We
have seen two codes agree while both were wrong. The solution of one model in
BioModels (MODEL1112050001, which is not curated) leaves the range of
double-precision numbers near t ≈ 3.4 × 10⁻⁷, but the simulation is requested to
t = 100. At the tolerances of our comparison, BNGsim and AMICI both kept a
species that should grow at an unstable equilibrium near zero, within the
absolute error tolerance, and continued to t = 100. Neither returned a
solution, and whether the two agreed depended on rounding
([PR #1547](https://github.com/lanl/bngsim/pull/1547)).

**Wrong answers look right.** A common failure is not a crash but a smooth,
plausible curve that is wrong. A software error has led to the retraction of
five papers (Miller, 2006). For this reason, BNGsim refuses constructs that it
knows it cannot simulate correctly, such as an algebraic rule, a `delay()` term,
or a fast reaction (one assumed to be at equilibrium), rather than drop the
construct and integrate a different system. Likewise, it refuses to report a
sensitivity that it knows it cannot compute correctly.

**Features interact.** Events, sensitivities, compartments that change size,
steady states, and stochastic simulation can each be tested alone. Many of the
defects we have fixed involved two features at once, such as a sensitivity
through an event that changes a compartment's size.

## What does BNGsim offer beyond SciPy and NumPy?

BNGsim returns its results as NumPy arrays (Harris et al., 2020). For an ODE
model written as a Python function, `scipy.integrate.solve_ivp` (Virtanen et
al., 2020) is the right tool. It offers implicit methods for stiff problems, but
it calls Python for each evaluation of the right-hand side, approximates the
Jacobian by finite differences unless the user supplies one, does not compute
sensitivities, and can only record an event or stop at one;
resetting the state and restarting are left to the user
([documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)).
BNGsim offers the following.

1. **A compiler for modeling languages.** BNGsim reads SBML, Antimony (Smith et
   al., 2009), and BNGL, derives the equations and their derivatives
   symbolically, and generates code that evaluates them. In physics, comparable
   tools include FEniCS and its Unified Form Language (Logg et al., 2012; Alnæs
   et al., 2014), Firedrake (Rathgeber et al., 2017), Devito (Louboutin et al.,
   2019), and Dedalus (Burns et al., 2020). In each, the user states the model
   symbolically, and the software generates the code that solves it.
2. **Sensitivities through discontinuities.** BNGsim computes forward
   sensitivities from analytic derivatives, including the jumps at events and at
   switches whose timing depends on a parameter, and including sensitivities to
   initial conditions set by initial assignments.
3. **Simulation without a reaction network.** For a rule-based model whose
   network is too large to generate, BNGsim runs NFsim (Sneddon et al., 2011)
   or RuleMonkey (Colvin et al., 2010). These are kinetic Monte Carlo methods in
   which rules act directly on molecules, which are represented as graphs. Their
   cost is independent of the number of reactions implied by the rules (Yang et
   al., 2008). They belong to the family of methods that includes the n-fold way
   of Bortz et al. (1975), Gillespie's method (Gillespie, 1976), and kinetic
   Monte Carlo in materials science (Voter, 2007).
4. **Verification as routine practice.** Every night, BNGsim's results are
   compared with results from BioNetGen, libRoadRunner, and AMICI, with
   finite-difference sensitivities, and with the SBML Test Suite, and a
   regression raises an alert. The other simulators are rerun weekly; on other
   nights, their recorded results are used
   ([how BNGsim is checked](benchmarks.md)). In the terms of Oberkampf and Roy
   (2010), this is evidence for code verification, that is, evidence that the
   code solves its equations correctly. Agreement with other codes is weaker
   evidence than agreement with exact solutions, which few models have.
   Validation, establishing that a model describes a biological system, is the
   modeler's task.
5. **A library meant to be embedded.** Once a model is loaded and, if needed,
   compiled, a simulation starts no subprocess and writes no result files.
   BNGsim's ODE, SSA, and PSA engines hold no global state (NFsim shares one
   random-number stream across the process), and BNGsim releases Python's global interpreter lock
   during a simulation, so batches run in parallel on threads. PyBioNetFit
   (Mitra et al., 2019) uses BNGsim in this way
   ([PyBNF guide](../user-guide/pybnf.md)).

## What BNGsim is not

BNGsim contributes no new numerical methods for ODEs. Integration is done by
CVODES or, optionally, by [Diffrax](https://github.com/patrick-kidger/diffrax)
through JAX, and steady states are found by integration, optionally refined with
KINSOL's Newton iteration. BNGsim does not compute adjoint sensitivities, which
give the gradient of an objective function at a cost nearly independent of the
number of parameters (Fröhlich et al., 2017b); AMICI does (Fröhlich et al., 2021). It does not solve
differential-algebraic equations (Brenan et al., 1995) or delay differential
equations (Bellen and Zennaro, 2003), and it refuses models that require them.
It has no graphical user interface; COPASI does (Hoops et al., 2006). For a
comparison of tools by task, see [When to use BNGsim](../choosing.md).

## References

- Alnæs, M.S., Logg, A., Ølgaard, K.B., Rognes, M.E., and Wells, G.N. (2014).
  Unified form language: a domain-specific language for weak formulations of
  partial differential equations. ACM Trans. Math. Softw. 40(2), 1–37.
  <https://doi.org/10.1145/2566630>
- Barr, E.T., Harman, M., McMinn, P., Shahbaz, M., and Yoo, S. (2015). The
  oracle problem in software testing: a survey. IEEE Trans. Softw. Eng. 41(5),
  507–525. <https://doi.org/10.1109/TSE.2014.2372785>
- Bellen, A., and Zennaro, M. (2003). Numerical Methods for Delay Differential
  Equations (Oxford University Press).
  <https://doi.org/10.1093/acprof:oso/9780198506546.001.0001>
- Bortz, A.B., Kalos, M.H., and Lebowitz, J.L. (1975). A new algorithm for Monte
  Carlo simulation of Ising spin systems. J. Comput. Phys. 17(1), 10–18.
  <https://doi.org/10.1016/0021-9991(75)90060-1>
- Brenan, K.E., Campbell, S.L., and Petzold, L.R. (1995). Numerical Solution of
  Initial-Value Problems in Differential-Algebraic Equations (SIAM).
  <https://doi.org/10.1137/1.9781611971224>
- Burns, K.J., Vasil, G.M., Oishi, J.S., Lecoanet, D., and Brown, B.P. (2020).
  Dedalus: a flexible framework for numerical simulations with spectral
  methods. Phys. Rev. Research 2(2), 023068.
  <https://doi.org/10.1103/PhysRevResearch.2.023068>
- Colvin, J., Monine, M.I., Gutenkunst, R.N., Hlavacek, W.S., Von Hoff, D.D.,
  and Posner, R.G. (2010). RuleMonkey: software for stochastic simulation of
  rule-based models. BMC Bioinformatics 11, 404.
  <https://doi.org/10.1186/1471-2105-11-404>
- Cortés, J. (2008). Discontinuous dynamical systems. IEEE Control Syst. Mag.
  28(3), 36–73. <https://doi.org/10.1109/MCS.2008.919306>
- Evans, T.W., Gillespie, C.S., and Wilkinson, D.J. (2008). The SBML discrete
  stochastic models test suite. Bioinformatics 24(2), 285–286.
  <https://doi.org/10.1093/bioinformatics/btm566>
- Faeder, J.R., Blinov, M.L., and Hlavacek, W.S. (2009). Rule-based modeling of
  biochemical systems with BioNetGen. Methods Mol. Biol. 500, 113–167.
  <https://doi.org/10.1007/978-1-59745-525-1_5>
- Filippov, A.F. (1988). Differential Equations with Discontinuous Righthand
  Sides (Kluwer). <https://doi.org/10.1007/978-94-015-7793-9>
- Fröhlich, F., Theis, F.J., Rädler, J.O., and Hasenauer, J. (2017a). Parameter
  estimation for dynamical systems with discrete events and logical operations.
  Bioinformatics 33(7), 1049–1056.
  <https://doi.org/10.1093/bioinformatics/btw764>
- Fröhlich, F., Kaltenbacher, B., Theis, F.J., and Hasenauer, J. (2017b).
  Scalable parameter estimation for genome-scale biochemical reaction networks.
  PLoS Comput. Biol. 13(1), e1005331.
  <https://doi.org/10.1371/journal.pcbi.1005331>
- Fröhlich, F., Weindl, D., Schälte, Y., Pathirana, D., Paszkowski, Ł., Lines,
  G.T., et al. (2021). AMICI: high-performance sensitivity analysis for large
  ordinary differential equation models. Bioinformatics 37(20), 3676–3677.
  <https://doi.org/10.1093/bioinformatics/btab227>
- Galán, S., Feehery, W.F., and Barton, P.I. (1999). Parametric sensitivity
  functions for hybrid discrete/continuous systems. Appl. Numer. Math. 31(1),
  17–47. <https://doi.org/10.1016/S0168-9274(98)00125-1>
- Gardner, D.J., Reynolds, D.R., Woodward, C.S., and Balos, C.J. (2022).
  Enabling new flexibility in the SUNDIALS suite of nonlinear and
  differential/algebraic equation solvers. ACM Trans. Math. Softw. 48(3), 1–24.
  <https://doi.org/10.1145/3539801>
- Gillespie, D.T. (1976). A general method for numerically simulating the
  stochastic time evolution of coupled chemical reactions. J. Comput. Phys.
  22(4), 403–434. <https://doi.org/10.1016/0021-9991(76)90041-3>
- Gillespie, D.T. (1977). Exact stochastic simulation of coupled chemical
  reactions. J. Phys. Chem. 81(25), 2340–2361.
  <https://doi.org/10.1021/j100540a008>
- Goebel, R., Sanfelice, R.G., and Teel, A.R. (2009). Hybrid dynamical systems.
  IEEE Control Syst. Mag. 29(2), 28–93.
  <https://doi.org/10.1109/MCS.2008.931718>
- Hairer, E., Nørsett, S.P., and Wanner, G. (1993). Solving Ordinary
  Differential Equations I: Nonstiff Problems, 2nd edn. (Springer).
  <https://doi.org/10.1007/978-3-540-78862-1>
- Hairer, E., and Wanner, G. (1996). Solving Ordinary Differential Equations II:
  Stiff and Differential-Algebraic Problems, 2nd edn. (Springer).
  <https://doi.org/10.1007/978-3-642-05221-7>
- Harris, C.R., Millman, K.J., van der Walt, S.J., Gommers, R., Virtanen, P.,
  Cournapeau, D., et al. (2020). Array programming with NumPy. Nature 585,
  357–362. <https://doi.org/10.1038/s41586-020-2649-2>
- Harris, L.A., Hogg, J.S., Tapia, J.-J., Sekar, J.A.P., Gupta, S., Korsunsky,
  I., et al. (2016). BioNetGen 2.2: advances in rule-based modeling.
  Bioinformatics 32(21), 3366–3368.
  <https://doi.org/10.1093/bioinformatics/btw469>
- Hatton, L. (1997). The T experiments: errors in scientific software. IEEE
  Comput. Sci. Eng. 4(2), 27–38. <https://doi.org/10.1109/99.609829>
- Hindmarsh, A.C., Brown, P.N., Grant, K.E., Lee, S.L., Serban, R., Shumaker,
  D.E., and Woodward, C.S. (2005). SUNDIALS: suite of nonlinear and
  differential/algebraic equation solvers. ACM Trans. Math. Softw. 31(3),
  363–396. <https://doi.org/10.1145/1089014.1089020>
- Hiskens, I.A., and Pai, M.A. (2000). Trajectory sensitivity analysis of hybrid
  systems. IEEE Trans. Circuits Syst. I 47(2), 204–220.
  <https://doi.org/10.1109/81.828574>
- Hlavacek, W.S., Faeder, J.R., Blinov, M.L., Perelson, A.S., and Goldstein, B.
  (2003). The complexity of complexes in signal transduction. Biotechnol.
  Bioeng. 84(7), 783–794. <https://doi.org/10.1002/bit.10842>
- Hlavacek, W.S., Faeder, J.R., Blinov, M.L., Posner, R.G., Hucka, M., and
  Fontana, W. (2006). Rules for modeling signal-transduction systems. Sci. STKE
  2006(344), re6. <https://doi.org/10.1126/stke.3442006re6>
- Hoops, S., Sahle, S., Gauges, R., Lee, C., Pahle, J., Simus, N., et al.
  (2006). COPASI—a COmplex PAthway SImulator. Bioinformatics 22(24),
  3067–3074. <https://doi.org/10.1093/bioinformatics/btl485>
- Hucka, M., Finney, A., Sauro, H.M., Bolouri, H., Doyle, J.C., Kitano, H., et
  al. (2003). The systems biology markup language (SBML): a medium for
  representation and exchange of biochemical network models. Bioinformatics
  19(4), 524–531. <https://doi.org/10.1093/bioinformatics/btg015>
- Hucka, M., Bergmann, F.T., Chaouiya, C., Dräger, A., Hoops, S., Keating,
  S.M., et al. (2019). The Systems Biology Markup Language (SBML): language
  specification for Level 3 Version 2 Core Release 2. J. Integr. Bioinform.
  16(2), 20190021. <https://doi.org/10.1515/jib-2019-0021>
- Kanewala, U., and Bieman, J.M. (2014). Testing scientific software: a
  systematic literature review. Inf. Softw. Technol. 56(10), 1219–1232.
  <https://doi.org/10.1016/j.infsof.2014.05.006>
- Keating, S.M., Waltemath, D., König, M., Zhang, F., Dräger, A., Chaouiya, C.,
  et al. (2020). SBML Level 3: an extensible format for the exchange and reuse
  of biological models. Mol. Syst. Biol. 16(8), e9110.
  <https://doi.org/10.15252/msb.20199110>
- Leine, R.I., and Nijmeijer, H. (2004). Dynamics and Bifurcations of
  Non-Smooth Mechanical Systems (Springer).
  <https://doi.org/10.1007/978-3-540-44398-8>
- Lin, Y.T., Feng, S., and Hlavacek, W.S. (2019). Scaling methods for
  accelerating kinetic Monte Carlo simulations of chemical reaction networks.
  J. Chem. Phys. 150(24), 244101. <https://doi.org/10.1063/1.5096774>
- Logg, A., Mardal, K.-A., and Wells, G.N., eds. (2012). Automated Solution of
  Differential Equations by the Finite Element Method: The FEniCS Book
  (Springer). <https://doi.org/10.1007/978-3-642-23099-8>
- Louboutin, M., Lange, M., Luporini, F., Kukreja, N., Witte, P.A., Herrmann,
  F.J., et al. (2019). Devito (v3.1.0): an embedded domain-specific language for
  finite differences and geophysical exploration. Geosci. Model Dev. 12(3),
  1165–1187. <https://doi.org/10.5194/gmd-12-1165-2019>
- Malik-Sheriff, R.S., Glont, M., Nguyen, T.V.N., Tiwari, K., Roberts, M.G.,
  Xavier, A., et al. (2020). BioModels—15 years of sharing computational models
  in life science. Nucleic Acids Res. 48(D1), D407–D415.
  <https://doi.org/10.1093/nar/gkz1055>
- McKeeman, W.M. (1998). Differential testing for software. Digit. Tech. J.
  10(1), 100–107.
- Meurer, A., Smith, C.P., Paprocki, M., Čertík, O., Kirpichev, S.B., Rocklin,
  M., et al. (2017). SymPy: symbolic computing in Python. PeerJ Comput. Sci. 3,
  e103. <https://doi.org/10.7717/peerj-cs.103>
- Miller, G. (2006). A scientist's nightmare: software problem leads to five
  retractions. Science 314(5807), 1856–1857.
  <https://doi.org/10.1126/science.314.5807.1856>
- Mitra, E.D., Suderman, R., Colvin, J., Ionkov, A., Hu, A., Sauro, H.M., et
  al. (2019). PyBioNetFit and the Biological Property Specification Language.
  iScience 19, 1012–1036. <https://doi.org/10.1016/j.isci.2019.08.045>
- Monniaux, D. (2008). The pitfalls of verifying floating-point computations.
  ACM Trans. Program. Lang. Syst. 30(3), 1–41.
  <https://doi.org/10.1145/1353445.1353446>
- Nagel, L.W. (1975). SPICE2: a computer program to simulate semiconductor
  circuits. Technical Report UCB/ERL M520, University of California, Berkeley.
  <https://www2.eecs.berkeley.edu/Pubs/TechRpts/1975/9602.html>
- Oberkampf, W.L., and Roy, C.J. (2010). Verification and Validation in
  Scientific Computing (Cambridge University Press).
  <https://doi.org/10.1017/CBO9780511760396>
- Rathgeber, F., Ham, D.A., Mitchell, L., Lange, M., Luporini, F., McRae,
  A.T.T., et al. (2017). Firedrake: automating the finite element method by
  composing abstractions. ACM Trans. Math. Softw. 43(3), 1–27.
  <https://doi.org/10.1145/2998441>
- Raue, A., Schilling, M., Bachmann, J., Matteson, A., Schelker, M., Kaschek,
  D., et al. (2013). Lessons learned from quantitative dynamical modeling in
  systems biology. PLoS ONE 8(9), e74335.
  <https://doi.org/10.1371/journal.pone.0074335>
- Serban, R., and Hindmarsh, A.C. (2005). CVODES: the sensitivity-enabled ODE
  solver in SUNDIALS. In Proceedings of the ASME 2005 International Design
  Engineering Technical Conferences, Vol. 6, 257–269.
  <https://doi.org/10.1115/DETC2005-85597>
- Shampine, L.F., and Gear, C.W. (1979). A user's view of solving stiff
  ordinary differential equations. SIAM Rev. 21(1), 1–17.
  <https://doi.org/10.1137/1021001>
- Shampine, L.F., and Thompson, S. (2000). Event location for ordinary
  differential equations. Comput. Math. Appl. 39(5–6), 43–54.
  <https://doi.org/10.1016/S0898-1221(00)00045-6>
- Shampine, L.F., Thompson, S., Kierzenka, J.A., and Byrne, G.D. (2005).
  Non-negative solutions of ODEs. Appl. Math. Comput. 170(1), 556–569.
  <https://doi.org/10.1016/j.amc.2004.12.011>
- Smith, L.P., Bergmann, F.T., Chandran, D., and Sauro, H.M. (2009). Antimony:
  a modular model definition language. Bioinformatics 25(18), 2452–2454.
  <https://doi.org/10.1093/bioinformatics/btp401>
- Sneddon, M.W., Faeder, J.R., and Emonet, T. (2011). Efficient modeling,
  simulation and coarse-graining of biological complexity with NFsim. Nat.
  Methods 8(2), 177–183. <https://doi.org/10.1038/nmeth.1546>
- Somogyi, E.T., Bouteiller, J.-M., Glazier, J.A., König, M., Medley, J.K.,
  Swat, M.H., et al. (2015). libRoadRunner: a high performance SBML simulation
  and analysis library. Bioinformatics 31(20), 3315–3321.
  <https://doi.org/10.1093/bioinformatics/btv363>
- Städter, P., Schälte, Y., Schmiester, L., Hasenauer, J., and Stapor, P.L.
  (2021). Benchmarking of numerical integration methods for ODE models of
  biological systems. Sci. Rep. 11, 2696.
  <https://doi.org/10.1038/s41598-021-82196-2>
- Vallabhajosyula, R.R., Chickarmane, V., and Sauro, H.M. (2006). Conservation
  analysis of large biochemical networks. Bioinformatics 22(3), 346–353.
  <https://doi.org/10.1093/bioinformatics/bti800>
- Virtanen, P., Gommers, R., Oliphant, T.E., Haberland, M., Reddy, T.,
  Cournapeau, D., et al. (2020). SciPy 1.0: fundamental algorithms for
  scientific computing in Python. Nat. Methods 17(3), 261–272.
  <https://doi.org/10.1038/s41592-019-0686-2>
- Voter, A.F. (2007). Introduction to the kinetic Monte Carlo method. In
  Radiation Effects in Solids, K.E. Sickafus, E.A. Kotomin, and B.P. Uberuaga,
  eds. (Springer), 1–23. <https://doi.org/10.1007/978-1-4020-5295-8_1>
- Welsh, C., Xu, J., Smith, L., König, M., Choi, K., and Sauro, H.M. (2023).
  libRoadRunner 2.0: a high performance SBML simulation and analysis library.
  Bioinformatics 39(1), btac770. <https://doi.org/10.1093/bioinformatics/btac770>
- Yang, J., Monine, M.I., Faeder, J.R., and Hlavacek, W.S. (2008). Kinetic Monte
  Carlo method for rule-based modeling of biochemical networks. Phys. Rev. E
  78(3), 031910. <https://doi.org/10.1103/PhysRevE.78.031910>
- Yang, X., Chen, Y., Eide, E., and Regehr, J. (2011). Finding and
  understanding bugs in C compilers. In Proceedings of the 32nd ACM SIGPLAN
  Conference on Programming Language Design and Implementation, 283–294.
  <https://doi.org/10.1145/1993498.1993532>
