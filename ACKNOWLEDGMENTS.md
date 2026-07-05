# Acknowledgments

BNGsim was developed and validated using a large body of excellent
open-source software from the systems-biology and scientific-Python
communities. Most of these tools are **not** redistributed with BNGsim. They
were used as reference implementations, parity oracles, library dependencies,
data corpora, or standards. We
gratefully acknowledge their authors and cite their work below.

> Third-party software and data that BNGsim redistributes (vendored code
> and model/test-data corpora, with their license terms) are listed separately in
> [`NOTICE`](NOTICE).

## Reference simulators and cross-validation oracles

BNGsim's numerical results were cross-validated against these simulation engines:

- **libRoadRunner** — the primary ODE and SSA parity oracle.
  - Somogyi ET, Bouteiller J-M, Glazier JA, König M, Medley JK, Swat MH,
    Sauro HM. libRoadRunner: a high performance SBML simulation and analysis
    library. *Bioinformatics.* 2015;31(20):3315–3321.
    doi:10.1093/bioinformatics/btv363
  - Welsh C, Xu J, Smith L, König M, Choi K, Sauro HM. libRoadRunner 2.0: a
    high-performance SBML simulation and analysis library. *Bioinformatics.*
    2023;39(1):btac770. doi:10.1093/bioinformatics/btac770
- **AMICI** — the reference for BNGsim's forward-sensitivity and gradient work.
  - Fröhlich F, Weindl D, Schälte Y, Pathirana D, Paszkowski Ł, Lines GT,
    Stapor P, Hasenauer J. AMICI: high-performance sensitivity analysis for
    large ordinary differential equation models. *Bioinformatics.*
    2021;37(20):3676–3677. doi:10.1093/bioinformatics/btab227
- **COPASI** — an independent third oracle (variable-volume, rate-rule, and
  stochastic cross-checks).
  - Hoops S, Sahle S, Gauges R, Lee C, Pahle J, Simus N, Singhal M, Xu L,
    Mendes P, Kummer U. COPASI—a COmplex PAthway SImulator. *Bioinformatics.*
    2006;22(24):3067–3074. doi:10.1093/bioinformatics/btl485
- **BioNetGen** (`BNG2.pl`, `run_network`, Network3) — used to generate reaction
  networks and reference trajectories, and as a rule-based ODE/SSA oracle.
  - Harris LA, Hogg JS, Tapia J-J, Sekar JAP, Gupta S, Korsunsky I, Arora A,
    Barua D, Sheehan RP, Faeder JR. BioNetGen 2.2: advances in rule-based
    modeling. *Bioinformatics.* 2016;32(21):3366–3368.
    doi:10.1093/bioinformatics/btw469
  - Blinov ML, Faeder JR, Goldstein B, Hlavacek WS. BioNetGen: software for
    rule-based modeling of signal transduction based on the interactions of
    molecular domains. *Bioinformatics.* 2004;20(17):3289–3291.
    doi:10.1093/bioinformatics/bth378
- **NFsim** — the network-free stochastic oracle (also vendored into BNGsim; see
  [`NOTICE`](NOTICE)).
  - Sneddon MW, Faeder JR, Emonet T. Efficient modeling, simulation and
    coarse-graining of biological complexity with NFsim. *Nat Methods.*
    2011;8(2):177–183. doi:10.1038/nmeth.1546

## Vendored simulation engines

BNGsim embeds these rule-based engines (license terms in [`NOTICE`](NOTICE)):

- **NFsim** — network-free simulation (cited above).
- **RuleMonkey** — the canonical rule/species-graph library and stochastic
  simulator. BNGsim vendors the current C++ version; the original simulator is
  described in:
  - Colvin J, Monine MI, Gutenkunst RN, Hlavacek WS, Von Hoff DD, Posner RG.
    RuleMonkey: software for stochastic simulation of rule-based models.
    *BMC Bioinformatics.* 2010;11:404. doi:10.1186/1471-2105-11-404

## Model-description standards and formats

BNGsim reads and/or writes these community standards:

- **SBML** (Systems Biology Markup Language) — core model interchange format.
  - Hucka M, Finney A, Sauro HM, Bolouri H, Doyle JC, Kitano H, et al. The
    systems biology markup language (SBML): a medium for representation and
    exchange of biochemical network models. *Bioinformatics.*
    2003;19(4):524–531. doi:10.1093/bioinformatics/btg015
  - Keating SM, Waltemath D, König M, et al. SBML Level 3: an extensible format
    for the exchange and reuse of biological models. *Mol Syst Biol.*
    2020;16(8):e9110. doi:10.15252/msb.20199110
- **SED-ML** (Simulation Experiment Description Markup Language) — read/write of
  simulation descriptions; also the format of the SED-ML protocol files BNGsim
  consumes from BioModels (via the `sys-bio/temp-biomodels` mirror).
  - Waltemath D, Adams R, Bergmann FT, Hucka M, Kolpakov F, Miller AK,
    Moraru II, Nickerson D, Sahle S, Snoep JL, Le Novère N. Reproducible
    computational biology experiments with SED-ML — the Simulation Experiment
    Description Markup Language. *BMC Syst Biol.* 2011;5:198.
    doi:10.1186/1752-0509-5-198
- **COMBINE archive / OMEX** — pack/unpack of SBML + SED-ML + manifest.
  - Bergmann FT, Adams R, Moodie S, Cooper J, Glont M, Golebiewski M, Hucka M,
    Laibe C, Miller AK, Nickerson DP, Olivier BG, Rodriguez N, Sauro HM,
    Scharm M, Soiland-Reyes S, Waltemath D, Yvon F, Le Novère N. COMBINE
    archive and OMEX format: one file to share all information to reproduce a
    modeling project. *BMC Bioinformatics.* 2014;15:369.
    doi:10.1186/s12859-014-0369-z
- **Antimony** — read/write via the Antimony library.
  - Smith LP, Bergmann FT, Chandran D, Sauro HM. Antimony: a modular model
    definition language. *Bioinformatics.* 2009;25(18):2452–2454.
    doi:10.1093/bioinformatics/btp401
- **BNGL** and the BNG `.net` format (BioNetGen, cited above), **Content
  MathML**, and **KiSAO** (the Kinetic Simulation Algorithm Ontology, used to
  tag SED-ML algorithms) are also handled.

## Numerical and infrastructure libraries

- **SUNDIALS** — the CVODES and KINSOL solvers underpin BNGsim's ODE
  integration, forward sensitivities, and steady-state solves (bundled in
  binary wheels).
  - Hindmarsh AC, Brown PN, Grant KE, Lee SL, Serban R, Shumaker DE,
    Woodward CS. SUNDIALS: Suite of nonlinear and differential/algebraic
    equation solvers. *ACM Trans Math Softw.* 2005;31(3):363–396.
    doi:10.1145/1089014.1089020
- **SuiteSparse / KLU** — sparse direct solver for the sparse-Jacobian path.
  - Davis TA, Palamadai Natarajan E. Algorithm 907: KLU, a direct sparse solver
    for circuit simulation problems. *ACM Trans Math Softw.*
    2010;37(3):Article 36. doi:10.1145/1824801.1824814
- **LAPACK** — dense linear algebra used by the numerical backends.
  - Anderson E, Bai Z, Bischof C, et al. LAPACK Users' Guide. 3rd ed.
    Philadelphia: SIAM; 1999. doi:10.1137/1.9780898719604
- **libSBML** — the SBML read/write API.
  - Bornstein BJ, Keating SM, Jouraku A, Hucka M. LibSBML: an API library for
    SBML. *Bioinformatics.* 2008;24(6):880–881.
    doi:10.1093/bioinformatics/btn051
- **NumPy** — array programming throughout BNGsim.
  - Harris CR, Millman KJ, van der Walt SJ, et al. Array programming with NumPy.
    *Nature.* 2020;585(7825):357–362. doi:10.1038/s41586-020-2649-2
- **SciPy** — scientific-computing routines (used in the test, benchmark, and
  parity suites).
  - Virtanen P, Gommers R, Oliphant TE, et al. SciPy 1.0: fundamental algorithms
    for scientific computing in Python. *Nat Methods.* 2020;17(3):261–272.
    doi:10.1038/s41592-019-0686-2
- **SymPy** — symbolic differentiation for code-generated sensitivities and
  Jacobians.
  - Meurer A, Smith CP, Paprocki M, et al. SymPy: symbolic computing in Python.
    *PeerJ Comput Sci.* 2017;3:e103. doi:10.7717/peerj-cs.103
- **pandas** — the `Result.to_dataframe()` export.
  - McKinney W. Data structures for statistical computing in Python. In:
    *Proceedings of the 9th Python in Science Conference (SciPy).* 2010:56–61.
    doi:10.25080/Majora-92bf1922-00a
- **ExprTk** (Arash Partow; https://github.com/ArashPartow/exprtk) — the
  vendored C++ math-expression evaluator; **MIR** (Vladimir Makarov;
  https://github.com/vnmakarov/mir) — the optional micro-JIT backend; and
  **pybind11** (Wenzel Jakob, Jason Rhinelander, Dean Moldovan;
  https://github.com/pybind/pybind11) — the C++/Python binding layer. These are
  software projects without an associated paper.

## Optional feature backends and interoperability

- **JAX** — the differentiable RHS/Jacobian bridge.
  - Bradbury J, Frostig R, Hawkins P, Johnson MJ, Katariya Y, Leary C,
    Maclaurin D, Necula G, Paszke A, VanderPlas J, Wanderman-Milne S, Zhang Q.
    JAX: composable transformations of Python+NumPy programs. 2018.
    http://github.com/jax-ml/jax
- **Diffrax** — the JAX-based differentiable ODE solver backend.
  - Kidger P. On Neural Differential Equations. PhD thesis, University of
    Oxford; 2021. arXiv:2202.02435
- **xarray** — labeled N-D arrays for the `Result.to_xarray()` / netCDF export.
  - Hoyer S, Hamman J. xarray: N-D labeled Arrays and Datasets in Python.
    *J Open Res Softw.* 2017;5(1):10. doi:10.5334/jors.148
- **h5py** — HDF5 archival of results (Andrew Collette and contributors;
  https://www.h5py.org).
- **Vivarium** — the `bngsim.vivarium` process wrapper for multiscale modeling.
  - Agmon E, Spangler RK, Skalnik CJ, Poole W, Peirce SM, Morrison JH,
    Covert MW. Vivarium: an interface and engine for integrative multiscale
    modeling in computational biology. *Bioinformatics.* 2022;38(7):1972–1979.
    doi:10.1093/bioinformatics/btac049
- **PyBNF (PyBioNetFit)** — the parameter-fitting front end that drives BNGsim,
  and the source of several example models.
  - Mitra ED, Suderman R, Colvin J, Ionkov A, Hu A, Sauro HM, Posner RG,
    Hlavacek WS. PyBioNetFit and the Biological Property Specification Language.
    *iScience.* 2019;19:1012–1036. doi:10.1016/j.isci.2019.08.045

## Test suites and model corpora

BNGsim's correctness is checked against, and its benchmarks draw from, these
community corpora (vendored corpora and their licenses are in [`NOTICE`](NOTICE)):

- **SBML Semantic Test Suite** — the SBML Team's conformance suite
  (https://github.com/sbmlteam/sbml-test-suite).
- **DSMTS (Discrete Stochastic Models Test Suite)** — vendored for hermetic
  stochastic testing.
  - Evans TW, Gillespie CS, Wilkinson DJ. The SBML discrete stochastic models
    test suite. *Bioinformatics.* 2008;24(2):285–286.
    doi:10.1093/bioinformatics/btm566
- **BioModels Database** — the curated ODE model corpus (with its SED-ML
  protocols) used for ODE benchmarking.
  - Malik-Sheriff RS, Glont M, Nguyen TVN, et al. BioModels—15 years of sharing
    computational models in life science. *Nucleic Acids Res.*
    2020;48(D1):D407–D415. doi:10.1093/nar/gkz1055
- **RuleHub** (RuleWorld; https://github.com/RuleWorld/RuleHub), **BNGL-Models**
  (https://github.com/wshlavacek/BNGL-Models), and the **BioNetGen model
  distribution** (Models2) — vendored BNGL model corpora (see [`NOTICE`](NOTICE)).

## Algorithms

- **Gillespie's Stochastic Simulation Algorithm (direct method)** — the basis of
  BNGsim's exact SSA.
  - Gillespie DT. Exact stochastic simulation of coupled chemical reactions.
    *J Phys Chem.* 1977;81(25):2340–2361. doi:10.1021/j100540a008
- **Extrande** — a thinning sampler used as an independent test oracle for the
  rate-rule-under-SSA hybrid path.
  - Voliotis M, Thomas P, Grima R, Bowsher CG. Stochastic simulation of
    biomolecular networks in dynamic environments. *PLoS Comput Biol.*
    2016;12(6):e1004923. doi:10.1371/journal.pcbi.1004923

## Build, test, and documentation tooling

BNGsim is built with **scikit-build-core**, **CMake**, **Ninja**, and
**pybind11** (stubs via **pybind11-stubgen**); tested with **pytest** and linted
with **ruff**, **mypy**, and **clang-format**; documented with **Sphinx**,
**MyST-Parser**, and **Furo**. Benchmark and parity tooling additionally uses
**matplotlib** and **bioservices**. We thank the maintainers of these projects.
