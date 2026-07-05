#!/usr/bin/env python3
"""Output-sensitivity cross-validation — BNGsim vs AMICI (+ FD guard).

Companion to ``run.py`` (which validates the *species* forward-sensitivity
tensor ``dx_species/dp``). This module validates the **output**-sensitivity
layer (GH #196–#207): ``Result.output_sensitivities()`` for

  * **observables** — ``d obs_j/dp = Σ_i c_ji · dx_i/dp`` (GH #197), and
  * **expressions** (BNGL functions) — the GH #198 chain rule
    ``d f/dp = ∂f/∂p + Σ_j (∂f/∂obs_j) · d obs_j/dp``.

It reuses ``run.py``'s alignment machinery wholesale: ``.net``→SBML export,
initial-condition / parameter seeding from the ``.net`` (the source of truth),
and the relerr kernel. The AMICI side: BNG observables AND functions are
emitted into the SBML as parameters targeted by ``assignmentRule``s, so
``amici.assignment_rules_to_observables`` turns the chosen ones into AMICI
observables and ``rdata.sy`` becomes the AD reference for ``d output/dp``.

One kind per model (see ``validate_model``): a function-bearing model
validates ``expression:`` selectors (registering the functions; their
observable references are exercised transitively); an observable-only model
validates ``observable:`` selectors. ``expr_demo`` is the function-bearing
model; the four signaling models are observable-only.

Two oracles per model:
  * **AMICI** ``sy`` — the primary cross-engine check (gated on max/p95).
  * **finite differences** on BNGsim's own output trajectories — an
    independent guard (gated on the median, since a single-step relative FD is
    inaccurate where a derivative is tiny relative to the output), so an AMICI
    alignment slip can't masquerade as agreement.

    python output_sens.py                 # all models
    python output_sens.py --model expr    # one model (expression path)
    python output_sens.py --no-fd         # skip the finite-difference guard
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

import bngsim

_SUITE_DIR = Path(__file__).resolve().parent
_BENCH_ROOT = _SUITE_DIR.parents[1]  # bngsim/benchmarks
sys.path.insert(0, str(_SUITE_DIR))
sys.path.insert(0, str(_BENCH_ROOT))

import run  # the forward_sens species runner — reuse its helpers  # noqa: E402

RESULTS_DIR = _SUITE_DIR / "results"
NET_DIR = _BENCH_ROOT / "models" / "net" / "ode"

# Models with observables, drawn from the run.py corpus + suite_ode horizons.
# (All four are observable-only — no BNGL functions — so this validates the
# observable chain rule, not yet the expression chain rule.)
MODELS = [
    # expr_demo is a small purpose-built model carrying BNGL functions
    # (satB, ratio) — the only one here that exercises the GH #198 EXPRESSION
    # output-sensitivity chain rule. The four signaling models are
    # observable-only.
    {"name": "expr_demo", "effort": "low"},
    {"name": "egfr_path", "effort": "low"},
    {"name": "tcr_signaling", "effort": "medium"},
    {"name": "Scaff_22_ground", "effort": "medium"},
    {"name": "SHP2_base_model", "effort": "high"},
]

# Cross-validation tolerances (mirror run.py's sensitivity xval philosophy:
# a relative noise floor keeps opposite-sign float noise on near-zero output
# sensitivities from dominating the headline relerr).
XVAL_RTOL = float(os.environ.get("S9_OUTSENS_RTOL", "1e-3"))
XVAL_ATOL_REL = float(os.environ.get("S9_OUTSENS_ATOL_REL", "1e-6"))
XVAL_ATOL = float(os.environ.get("S9_OUTSENS_ATOL", "1e-10"))


# ── AMICI API shims (this build is snake_case; older builds camelCase) ──────


def _call(obj, camel, snake, *a):
    fn = getattr(obj, camel, None) or getattr(obj, snake, None)
    if fn is None:
        raise AttributeError(f"neither {camel} nor {snake} on {type(obj)}")
    return fn(*a)


def _first(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return list(getattr(obj, n)())
    raise AttributeError(f"none of {names} on {type(obj)}")


def _amici_run(amici, amici_sundials, model, solver):
    if hasattr(amici, "runAmiciSimulation"):
        return amici.runAmiciSimulation(model, solver)
    if hasattr(amici_sundials, "run_simulation"):
        return amici_sundials.run_simulation(model, solver)
    return model.simulate(solver)


def _build_amici_observables(sbml_path, build_dir, register_names, t_end, n_points, net_path):
    """Compile SBML→AMICI with the named BNG assignment rules registered as
    AMICI observables, seeded from the ``.net``.

    ``register_names`` are the SBML assignment-rule targets to convert into
    AMICI observables — BNG observables *or* functions (both emit as
    assignment-rule parameters). For the expression path, pass only the
    function names so the observables they reference stay as model
    expressions (AMICI substitutes through them when computing ``sy``).

    Returns dict with the AMICI ``sy`` (transposed to ``(nt, n_out, n_param)``),
    observable ids, parameter ids, and per-id parameter values.
    """
    import amici

    amici_rt, amici_sundials = run.prepare_amici_runtime()
    importer = amici.SbmlImporter(str(sbml_path))
    wanted = set(register_names)
    channels = amici.assignment_rules_to_observables(
        importer.sbml_model,
        filter_function=lambda p: p.getId() in wanted,
        as_dict=False,
    )
    model_name = Path(sbml_path).stem.replace("-", "_").replace(".", "_") + "_outsens"
    importer.sbml2amici(model_name, str(build_dir), observation_model=channels)

    module = amici.import_model_module(model_name, str(build_dir))
    model = module.getModel() if hasattr(module, "getModel") else module.get_model()
    _call(model, "setTimepoints", "set_timepoints", np.linspace(0, t_end, n_points))

    run.seed_amici_initial_state_from_net(model, Path(net_path))
    run.seed_amici_parameters_from_net(model, Path(net_path))

    solver = _call(model, "getSolver", "create_solver")
    _call(solver, "setAbsoluteTolerance", "set_absolute_tolerance", 1e-12)
    _call(solver, "setRelativeTolerance", "set_relative_tolerance", 1e-10)
    if hasattr(solver, "setSensitivityMethod"):
        solver.setSensitivityMethod(amici_rt.SensitivityMethod.forward)
        solver.setSensitivityOrder(amici_rt.SensitivityOrder.first)
    else:
        solver.set_sensitivity_method(amici_sundials.SensitivityMethod_forward)
        solver.set_sensitivity_order(amici_sundials.SensitivityOrder_first)

    nparam = int(model.np() if callable(getattr(model, "np", None)) else model.np)
    if hasattr(model, "setParameterScale"):
        model.setParameterScale([amici_rt.ParameterScaling.none] * nparam)
    else:
        model.set_parameter_scale([amici_sundials.ParameterScaling_none] * nparam)

    param_ids = _first(model, "get_free_parameter_ids", "getParameterIds", "get_parameter_ids")
    obs_ids = _first(model, "get_observable_ids", "getObservableIds")
    rdata = _amici_run(amici_rt, amici_sundials, model, solver)
    sy = np.asarray(rdata.sy)
    # AMICI emits sy as (nt, n_param, n_obs); bngsim is (nt, n_obs, n_param).
    if sy.ndim == 3 and sy.shape[1] == len(param_ids) and sy.shape[2] == len(obs_ids):
        sy = np.transpose(sy, (0, 2, 1))
    _, _, pvals_by_id = run._amici_all_parameter_ids_values(model)
    return {
        "sy": sy,
        "obs_ids": obs_ids,
        "param_ids": param_ids,
        "param_values_by_id": pvals_by_id,
    }


def _bngsim_fd_output_sens(net_path, out_names, params, t_end, n_points, selector_prefix):
    """Central-difference d(output)/dp on BNGsim's own output trajectories.

    Independent of AMICI; uses a relative step so tiny rate constants are
    perturbed sensibly. ``selector_prefix`` is ``"observable"`` or
    ``"expression"``. Returns ``(nt, n_out, n_param)``.
    """
    sel = [f"{selector_prefix}:{n}" for n in out_names]
    p0 = run._bngsim_param_value_map(str(net_path))

    def _traj(overrides):
        m = bngsim.Model.from_net(str(net_path))
        run._apply_param_overrides(m, overrides)
        r = bngsim.Simulator(m, method="ode", net_path=str(net_path)).run(
            t_span=(0, t_end), n_points=n_points
        )
        return np.asarray(r.outputs(sel))

    cols = []
    for p in params:
        base = float(p0[p])
        eps = 1e-6 * abs(base) if base != 0 else 1e-9
        hi = _traj({p: base + eps})
        lo = _traj({p: base - eps})
        cols.append((hi - lo) / (2 * eps))
    return np.stack(cols, axis=-1)


def _xval_block(bng_tensor, ref_tensor, *, label):
    """Noise-floored relerr stats between two ``(nt, n_obs, n_param)`` tensors."""
    nt = min(bng_tensor.shape[0], ref_tensor.shape[0])
    a = ref_tensor[:nt]
    b = bng_tensor[:nt]
    scale = float(max(np.abs(a).max() if a.size else 0.0, np.abs(b).max() if b.size else 0.0))
    atol = max(XVAL_ATOL, XVAL_ATOL_REL * scale)
    stats = run._relerr_stats(b, a, atol=atol)
    stats["atol_eff"] = atol
    # AMICI is exact AD — gate on max/p95. FD is an inherently approximate
    # guard (a single relative step is inaccurate where a derivative is tiny
    # relative to the output), so gate it on the MEDIAN: a real obs/param
    # *alignment* bug corrupts every cell and blows up the median, while
    # near-zero-derivative tail noise leaves it untouched.
    if label == "fd":
        stats["pass"] = bool(stats["med"] <= XVAL_RTOL)
    else:
        stats["pass"] = bool(stats["max"] <= XVAL_RTOL or stats["p95"] <= XVAL_RTOL)
    stats["label"] = label
    return stats


def _validate_kind(r, bng_params, net_path, sbml, build_dir, kind, out_names, t_end, n_points, do_fd):
    """Cross-validate one output kind (``observable`` or ``expression``).

    Builds an AMICI model with ``out_names`` registered as observables,
    compares bngsim ``output_sensitivities("<kind>:...")`` against AMICI ``sy``
    (primary) and a finite-difference guard (median-gated). Returns a result
    dict or ``None`` if there's nothing alignable.
    """
    try:
        info = _build_amici_observables(sbml, build_dir, out_names, t_end, n_points, net_path)
    except Exception as e:  # AMICI compile/import can fail (e.g. dangling refs)
        return {"skip_reason": f"amici build failed: {str(e)[:160]}"}
    am_out = info["obs_ids"]
    common_out = [o for o in am_out if o in set(out_names)]
    am_pidx = {p: i for i, p in enumerate(info["param_ids"])}
    bn_pidx = {p: i for i, p in enumerate(bng_params)}
    common_params = [p for p in info["param_ids"] if p in bn_pidx]
    if not common_out or not common_params:
        return None

    bos = np.asarray(r.output_sensitivities([f"{kind}:{o}" for o in common_out]))
    b_out_ix = {o: i for i, o in enumerate(common_out)}
    bi = [bn_pidx[p] for p in common_params]
    B = bos[:, [b_out_ix[o] for o in common_out], :][:, :, bi]

    am_out_ix = {o: i for i, o in enumerate(am_out)}
    ai = [am_pidx[p] for p in common_params]
    A = info["sy"][:, [am_out_ix[o] for o in common_out], :][:, :, ai]

    res = {"n_outputs": len(common_out), "n_params": len(common_params)}
    res["amici"] = _xval_block(B, A, label="amici")
    if do_fd:
        FD = _bngsim_fd_output_sens(net_path, common_out, common_params, t_end, n_points, kind)
        res["fd"] = _xval_block(B, FD, label="fd")
    res["pass"] = bool(res["amici"]["pass"] and (not do_fd or res["fd"]["pass"]))
    return res


def validate_model(cfg, suite_by_name, *, do_fd=True):
    """Cross-validate observable (and, if present, expression) output
    sensitivities for one model against AMICI + an FD guard."""
    name = cfg["name"]
    net_path = NET_DIR / f"{name}.net"
    out = {"name": name, "status": "ok"}
    if not net_path.exists():
        return {**out, "status": "skip", "reason": f".net not found: {net_path}"}
    bngl = run._find_bngl(name)
    if bngl is None:
        return {**out, "status": "skip", "reason": "no companion .bngl"}

    sm = suite_by_name.get(name, {})
    t_end = float(sm.get("t_end", 100))
    n_points = int((sm.get("n_steps") or sm.get("n_points") or 200)) + 1

    m = bngsim.Model.from_net(str(net_path))
    obs_names = list(m.observable_names)
    params = list(run.get_param_names(str(net_path)))
    sim = bngsim.Simulator(m, method="ode", sensitivity_params=params, net_path=str(net_path))
    r = sim.run(t_span=(0, t_end), n_points=n_points)
    bng_params = list(r.sensitivity_params)
    expr_names = list(r.expression_names)  # BNGL functions (GH #198), if any

    # One kind per model. A function-bearing model validates the EXPRESSION
    # kind: BNG functions reference observables, so converting the observables
    # to AMICI observables (which removes their assignment rules) would leave
    # the function rules dangling and break AMICI's C++ codegen — and the
    # expression chain rule (e.g. satB = Btot/(Km+Btot)) exercises the
    # observable sensitivity transitively anyway. Observable-only models
    # validate the OBSERVABLE kind. (The four signaling models cover
    # observables directly; expr_demo covers expressions.)
    if expr_names:
        kinds = [("expression", expr_names)]
    elif obs_names:
        kinds = [("observable", obs_names)]
    else:
        return {**out, "status": "skip", "reason": "model has no observables or functions"}

    with tempfile.TemporaryDirectory(prefix=f"outsens_{name}_") as tmp:
        sbml = run._convert_bngl_to_sbml(str(bngl), tmp)
        if sbml is None:
            return {**out, "status": "skip", "reason": "SBML export failed"}
        out["kinds"] = {}
        skip_reasons = []
        for kind, out_names in kinds:
            res = _validate_kind(
                r, bng_params, net_path, sbml, Path(tmp) / f"build_{kind}",
                kind, out_names, t_end, n_points, do_fd,
            )
            if res is None:
                skip_reasons.append(f"{kind}: no alignable outputs/params")
            elif "pass" not in res:
                skip_reasons.append(f"{kind}: {res.get('skip_reason', 'skipped')}")
            else:
                out["kinds"][kind] = res

    if not out["kinds"]:
        return {**out, "status": "skip", "reason": "; ".join(skip_reasons) or "no kinds"}
    out["pass"] = bool(all(k["pass"] for k in out["kinds"].values()))
    return out


def main():
    ap = argparse.ArgumentParser(description="Observable output-sensitivity xval: BNGsim vs AMICI")
    ap.add_argument("--model", type=str, default="", help="Run only this model (substring match)")
    ap.add_argument("--no-fd", action="store_true", help="Skip the finite-difference guard")
    from _effort import add_effort_arg, filter_by_effort

    add_effort_arg(ap)
    args = ap.parse_args()

    models = filter_by_effort(MODELS, args.effort, key=lambda m: m["effort"])
    if args.model:
        models = [m for m in models if args.model.lower() in m["name"].lower()]

    suite_by_name = {m["name"]: m for m in run.load_suite(run.SUITE_ODE)}

    print("=" * 78)
    print("  Output-sensitivity xval — BNGsim vs AMICI (+ FD guard)")
    print("  (observable: + expression: selectors; one kind per model)")
    print("=" * 78)

    results = []
    for cfg in models:
        print(f"\n--- {cfg['name']} ---", flush=True)
        try:
            res = validate_model(cfg, suite_by_name, do_fd=not args.no_fd)
        except Exception as e:  # pragma: no cover - per-model robustness
            res = {"name": cfg["name"], "status": "error", "error": str(e)[:300]}
        results.append(res)
        if res["status"] == "skip":
            print(f"  SKIP: {res['reason']}")
        elif res["status"] == "error":
            print(f"  ERROR: {res['error']}")
        else:
            for kind, k in res["kinds"].items():
                a = k["amici"]
                print(
                    f"  [{kind}] n={k['n_outputs']} params={k['n_params']}  "
                    f"AMICI: {'PASS' if a['pass'] else 'FAIL'} "
                    f"[max={a['max']:.2e} p95={a['p95']:.2e} med={a['med']:.2e}]"
                )
                if "fd" in k:
                    f = k["fd"]
                    print(
                        f"        FD guard: {'PASS' if f['pass'] else 'FAIL'} "
                        f"(bulk med={f['med']:.2e}; tail p95={f['p95']:.2e} max={f['max']:.2e} "
                        f"— tail near-zero-derivative FD noise, gated on median)"
                    )

    print("\n" + "=" * 78)
    ok = [r for r in results if r["status"] == "ok"]
    npass = sum(1 for r in ok if r.get("pass"))
    print(f"  SUMMARY: {npass}/{len(ok)} models PASS output-sensitivity xval")

    payload = {"machine_info": run.get_machine_info(), "results": results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "output_sens_results.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  Results: {path}")


if __name__ == "__main__":
    main()
