---
summary: "Known limitations: where to find the open defects that can give a wrong answer with no error, what BNGsim refuses, where it departs from other simulators on purpose, and what is known about each disagreement in the published benchmarks."
---

# Known limitations

This page collects what can stand between a model and a right answer. There
are three kinds, and they differ in what you see:

- **A refusal** is an error. BNGsim stops and names what it cannot do.
- **A departure on purpose** gives a different answer from another simulator,
  for a documented reason.
- **An open defect** can give a wrong answer with no error. Check whether your
  model uses the feature it names.

## Open defects

The list changes as defects are found and fixed, so it is kept on the issue
tracker and not copied here: the
[open issues labeled `correctness concern`](https://github.com/lanl/bngsim/issues?q=is%3Aissue+is%3Aopen+label%3A%22correctness+concern%22)
are the defects known to give a plausible wrong result with no error. Most
titles name the feature, an input that goes wrong and by how much. Most need a
particular combination, such as a sensitivity through an event at the same
instant as another event. Read the titles for the features your model uses:
events, switches on time or on a species, table functions, the network-free
methods, sensitivities, steady states, JAX and `run_diffrax`, the converters
(`sbml_to_net`, `sbml_to_bngl`), compiled against interpreted expressions, and
the SBML constructs your model relies on.

That list describes the development version. An open issue is not fixed in
any release, the newest included. An issue is closed when its fix is merged, so a closed one may still
affect the release you have: the
[changelog](https://github.com/lanl/bngsim/blob/HEAD/CHANGELOG.md) says, release
by release, what was fixed and for whom.

`result.check("all")` runs a model again on other solver paths, and in
libRoadRunner where it can. It catches some defects on your own model, not
all. See [Check a result before you rely on it](../user-guide/checking.md).

## What BNGsim refuses

BNGsim refuses what it cannot simulate faithfully, with an error that names
the construct, rather than simulate a different system. The main cases:

- **SBML** algebraic rules, delays and fast reactions, which need a solver
  BNGsim does not have, and required SBML packages it cannot read. The list,
  and the trivial forms that still load, is under
  [SBML: what BNGsim refuses, and why](benchmarks.md#sbml-what-bngsim-refuses-and-why).
- **Sensitivities** a model's math does not allow, refused with a
  `SensitivityUnsupportedError` that names the construct. See the
  [error codes](../reference/error-codes.md).
- **`method="rm"` (RuleMonkey, also called `"nf_exact"`)** cannot change a
  parameter once its session has started. A protocol that changes one between
  two network-free steps raises a `ParameterError`. `method="nf"` (NFsim) can.

## Departures on purpose

- **Avogadro's number.** BNGsim uses the SI value, exactly 6.02214076e23 per
  mole, for SBML's `avogadro` and for BNGL's `_NA`. The SBML Test Suite grades
  three cases (00960, 00961 and 01323) against the older 6.02214179e23, so
  BNGsim differs from them by 1.7e-7 of the value.
- **SBML events at the same instant with the same priority** are ordered by a
  random draw, as SBML Level 3 says, from a fixed default seed, so every run is
  reproducible. Pass `seed=` for another ordering.

[Benchmarks & validation](benchmarks.md) has the details of both.

## Disagreements in the published benchmarks

[bngsim-benchmarks](https://github.com/lanl/bngsim-benchmarks/tree/main/published)
compares BNGsim with other simulators on published models. A `DIFF` there
means that BNGsim and a reference disagree. By itself it does not say which
one is wrong. The table says what is known about each disagreement, and each
BNGsim failure, in the snapshots measured on BNGsim 0.17.2 (published
2026-11-25). Where a
reference is said to be wrong, the evidence is the developer parity suite's
analysis of the same model, on the engine versions it names:

| suite | model | reference | what is known |
|---|---|---|---|
| ode-sbml | brain_energy_metabolism_winter2017 (BIOMD0000000627) | libRoadRunner | **The reference is wrong.** Blood flow rises on a ramp centred at t = 199, a second before the stimulus turns on. libRoadRunner steps over the ramp. With its step bounded at 0.05 and at 0.01, it agrees with itself, and with BNGsim to about 4e-7 ([#482](https://github.com/lanl/bngsim/issues/482), whose first comment has the triage). |
| ode-sbml | brain_energy_metabolism_winter2017 | COPASI | **The reference is wrong.** COPASI steps over the same ramp and never applies the stimulus, which is on from t = 200 to t = 240. A one-species model with the same window reproduces it. With its step bounded, COPASI agrees with BNGsim ([#1563](https://github.com/lanl/bngsim/issues/1563)). |
| ode-sbml | insulin_signalling_oxidative_stress_smith2013 | AMICI | **Cause not established.** AMICI's result is far from BNGsim's (by about 5e219 of a species' peak), while BNGsim agrees with run_network and libRoadRunner on this model. |
| nf-xml-signaling | tcr_signaling_chylek2014 and erbb_receptor_signaling_creamer2012, the `nf_exact` setting | none (outcome `EXCEPTION`: BNGsim's own run raised) | **A refusal.** Each protocol changes a parameter between network-free steps, which RuleMonkey refuses (above). |
| sens-net | cstar_trka_ngf_rukhlenko2022 and cstar_trkb_bdnf_rukhlenko2022 | none (outcome `UNSUPPORTED`) | **A refusal.** BNGsim will not carry forward sensitivities across the species reset that starts the protocol's second phase. |
| nf-xml-aggregation, nf-xml-signaling | 10 `DIFF` jobs on 6 models: tlbr_solution_macken1982, blbr_cooperativity_posner2004, egfr_oligomerization_mitra2019, p53_nhej_dolan2015, car_cd3zeta_phosphorylation_rohrs2018, and erbb_receptor_signaling_creamer2012 (its `nf_reject` setting) | NFsim, RuleMonkey | **Not assessed yet.** The suites' harness and verdicts changed after these snapshots were taken. The rows will be judged again when the suites are next run. |

The other outcomes in those reports, `REFERENCE_FAILED` and `refuted`, are
findings about a reference, not about BNGsim. Each report says why.
