#!/usr/bin/env python3
"""Minimal bounded ODE fitting demo from a BNG .net file.

This script demonstrates the same workflow shown in the manuscript code box:

1) Parse a .net file with the universal parser (pure Python),
2) Build a BNGsim model from parsed data,
3) Run ODE simulation with forward sensitivities,
4) Fit parameters with SciPy trust-region reflective least squares.

By default, synthetic data are generated from the input model's default
parameter values (optionally with Gaussian noise), so the script is runnable
out of the box.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import bngsim
import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[3]  # repo root (the bngsim/ tree)
DEFAULT_NET = REPO_ROOT / "tests" / "data" / "simple_decay.net"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results" / "ode_trf_fit_from_net"


@dataclass
class FitSummary:
    net_file: str
    parameter_names: list[str]
    true_params: list[float]
    initial_guess: list[float]
    lower_bounds: list[float]
    upper_bounds: list[float]
    estimated_params: list[float]
    max_abs_param_error: float
    nfev: int
    njev: int | None
    cost: float
    success: bool
    status: int
    message: str
    t_start: float
    t_end: float
    n_points: int
    noise_std: float
    data_source: str


def _parse_vector(text: str, n: int, name: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 1:
        return np.full(n, float(parts[0]), dtype=float)
    if len(parts) != n:
        raise ValueError(f"{name} must contain either 1 value or {n} values (got {len(parts)}).")
    return np.array([float(p) for p in parts], dtype=float)


def _resolve_params(model: bngsim.Model, names_arg: str) -> list[str]:
    if not names_arg.strip():
        return list(model.param_names)
    names = [p.strip() for p in names_arg.split(",") if p.strip()]
    known = set(model.param_names)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"Unknown parameter(s): {unknown}. Known: {model.param_names}")
    return names


def _load_or_generate_data(
    model: bngsim.Model,
    t_start: float,
    t_end: float,
    n_points: int,
    data_path: Path | None,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, str]:
    if data_path is not None:
        arr = np.load(data_path)
        if arr.ndim != 2:
            raise ValueError(f"Observed data must be 2D (n_times, n_species), got {arr.shape}.")
        if arr.shape[0] != n_points:
            raise ValueError(
                f"Observed data n_times ({arr.shape[0]}) does not match n_points ({n_points})."
            )
        if arr.shape[1] != model.n_species:
            raise ValueError(
                f"Observed data n_species ({arr.shape[1]}) does not match model ({model.n_species})."
            )
        return np.asarray(arr, dtype=float), f"file:{data_path}"

    sim = bngsim.Simulator(model, method="ode")
    model.reset()
    result = sim.run(t_span=(t_start, t_end), n_points=n_points)
    data = np.asarray(result.species, dtype=float)
    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        data = data + rng.normal(0.0, noise_std, size=data.shape)
    return data, "synthetic"


def run_fit(args: argparse.Namespace) -> FitSummary:
    net_file = args.net.resolve()
    if not net_file.exists():
        raise FileNotFoundError(f"Net file not found: {net_file}")

    parsed = bngsim.parse_net_file(str(net_file))
    model = bngsim.build_model_from_parsed(parsed)
    pnames = _resolve_params(model, args.params)
    n_params = len(pnames)
    if n_params == 0:
        raise ValueError("No parameters selected for fitting.")

    true_p = np.array([model.get_param(n) for n in pnames], dtype=float)

    if args.x0.strip():
        x0 = _parse_vector(args.x0, n_params, "x0")
    else:
        x0 = np.maximum(true_p * 2.0, 1e-8)

    lb = _parse_vector(args.lower, n_params, "lower")
    ub = _parse_vector(args.upper, n_params, "upper")
    if np.any(lb >= ub):
        raise ValueError("All lower bounds must be strictly less than upper bounds.")

    observed, data_source = _load_or_generate_data(
        model=model,
        t_start=args.t_start,
        t_end=args.t_end,
        n_points=args.n_points,
        data_path=args.data,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    sim = bngsim.Simulator(model, method="ode", sensitivity_params=pnames)
    cache: dict[str, np.ndarray | None] = {"p": None, "r": None, "J": None}

    def eval_model(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cached_p = cache["p"]
        if cached_p is not None and np.array_equal(p, cached_p):
            return cache["r"], cache["J"]  # type: ignore[return-value]

        model.set_params(dict(zip(pnames, p, strict=False)))
        model.reset()
        out = sim.run(t_span=(args.t_start, args.t_end), n_points=args.n_points)

        resid = (out.species - observed).ravel()
        jac = out.sensitivities.reshape(-1, n_params)
        cache["p"] = p.copy()
        cache["r"] = resid
        cache["J"] = jac
        return resid, jac

    fit = least_squares(
        fun=lambda p: eval_model(p)[0],
        jac=lambda p: eval_model(p)[1],
        x0=x0,
        bounds=(lb, ub),
        method="trf",
    )

    est = np.asarray(fit.x, dtype=float)
    max_abs_err = float(np.max(np.abs(est - true_p)))

    return FitSummary(
        net_file=str(net_file),
        parameter_names=pnames,
        true_params=true_p.tolist(),
        initial_guess=x0.tolist(),
        lower_bounds=lb.tolist(),
        upper_bounds=ub.tolist(),
        estimated_params=est.tolist(),
        max_abs_param_error=max_abs_err,
        nfev=int(fit.nfev),
        njev=None if fit.njev is None else int(fit.njev),
        cost=float(fit.cost),
        success=bool(fit.success),
        status=int(fit.status),
        message=str(fit.message),
        t_start=float(args.t_start),
        t_end=float(args.t_end),
        n_points=int(args.n_points),
        noise_std=float(args.noise_std),
        data_source=data_source,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run bounded ODE parameter fitting from a BNG .net file."
    )
    p.add_argument("--net", type=Path, default=DEFAULT_NET, help="Path to .net file.")
    p.add_argument(
        "--params",
        type=str,
        default="",
        help="Comma-separated fit parameter names. Default: all model parameters.",
    )
    p.add_argument("--t-start", type=float, default=0.0, help="Simulation start time.")
    p.add_argument("--t-end", type=float, default=10.0, help="Simulation end time.")
    p.add_argument("--n-points", type=int, default=101, help="Number of simulation points.")
    p.add_argument(
        "--x0",
        type=str,
        default="",
        help="Initial guess vector (comma-separated). Default: 2x true parameters.",
    )
    p.add_argument(
        "--lower",
        type=str,
        default="1e-12",
        help="Lower bounds (single value or comma-separated vector).",
    )
    p.add_argument(
        "--upper",
        type=str,
        default="1e3",
        help="Upper bounds (single value or comma-separated vector).",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional .npy observed data (shape n_times x n_species).",
    )
    p.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std for synthetic data generation.",
    )
    p.add_argument("--seed", type=int, default=1, help="RNG seed for synthetic noise.")
    p.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Directory for fit summary JSON.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_fit(args)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_json = args.outdir / "results.json"
    out_json.write_text(json.dumps(asdict(summary), indent=2))

    print("=== ODE TRF FIT SUMMARY ===")
    print(f"net_file: {summary.net_file}")
    print(f"parameters: {summary.parameter_names}")
    print(f"true_params: {summary.true_params}")
    print(f"initial_guess: {summary.initial_guess}")
    print(f"estimated_params: {summary.estimated_params}")
    print(f"max_abs_param_error: {summary.max_abs_param_error:.6g}")
    print(f"success: {summary.success} (status={summary.status})")
    print(f"message: {summary.message}")
    print(f"cost: {summary.cost:.6g}")
    print(f"nfev/njev: {summary.nfev}/{summary.njev}")
    print(f"results_json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
