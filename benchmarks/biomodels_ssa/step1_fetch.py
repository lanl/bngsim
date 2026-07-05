#!/usr/bin/env python3
"""Fetch models from BioModels database.

Downloads SBML files from the EBI BioModels repository.
Automatically avoids duplicates by tracking fetch history.
By default, only ODE models are fetched (~1,680 models).

Adapted from ssys biomodels_batch/step1_fetch.py.

Usage:
    # Fetch all available ODE models
    python step1_fetch.py --target-total 2000

    # Fetch 100 random ODE models
    python step1_fetch.py --n 100 --strategy random

    # Fetch ALL models including non-ODE
    python step1_fetch.py --n 500 --all-models
"""

import argparse
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import config
import utils
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FetchResult(TypedDict):
    """Result of fetching a single model."""

    model_id: str
    sbml_success: bool
    error: str | None


def load_fetch_history() -> dict:
    """Load fetch history from JSON file."""
    history_path = Path(config.FETCH_HISTORY)
    if history_path.exists():
        try:
            with open(history_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupted fetch history: {e}, starting fresh")

    return {
        "fetch_sessions": [],
        "total_unique_models": 0,
        "all_fetched_ids": [],
    }


def save_fetch_history(history: dict):
    """Save fetch history to JSON file."""
    history_path = Path(config.FETCH_HISTORY)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def get_already_fetched_ids() -> set[str]:
    """Return set of all model IDs already fetched."""
    history = load_fetch_history()
    return set(history.get("all_fetched_ids", []))


def filter_ode_models(all_models: list[str]) -> list[str]:
    """Filter model list to only include ODE models.

    Checks each model's metadata to determine if it's an
    ODE model by querying the BioModels API for
    modellingApproach.
    """
    from bioservices import BioModels

    bm = BioModels()

    ode_models = []
    logger.info(f"Filtering for ODE models from {len(all_models)} total...")

    for i, model_id in enumerate(all_models):
        if (i + 1) % 100 == 0:
            logger.info(
                f"  Checked {i + 1}/{len(all_models)} models, "
                f"found {len(ode_models)} ODE models so far..."
            )

        try:
            info = bm.get_model(model_id)

            if not info or not isinstance(info, dict):
                continue

            approach = info.get("modellingApproach", {})
            if isinstance(approach, dict):
                approach_name = approach.get("name", "").lower()
            else:
                approach_name = str(approach).lower()

            is_ode = "ordinary differential equation" in approach_name

            if is_ode:
                ode_models.append(model_id)

            time.sleep(0.05)

        except Exception as e:
            logger.debug(f"Could not check model {model_id}: {e}")
            continue

    logger.info(f"Found {len(ode_models)} ODE models out of {len(all_models)} total")
    return ode_models


def query_available_models(
    ode_only: bool = True,
) -> list[str]:
    """Get available model IDs from BioModels.

    Caches result in model_registry.json.
    """
    registry_path = Path(config.MODEL_REGISTRY)
    cache_key = "ode_models" if ode_only else "models"

    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
            cache_age = time.time() - registry.get("timestamp", 0)
            if cache_age < 7 * 24 * 3600 and cache_key in registry:
                n = len(registry[cache_key])
                logger.info(f"Using cached model registry ({n} models)")
                return registry[cache_key]

    logger.info("Querying BioModels for available models...")
    try:
        from bioservices import BioModels

        bm = BioModels()

        models = []
        offset = 0
        batch_size = 100

        logger.info("Fetching model list with pagination...")
        while True:
            search_result = bm.search("*", offset=offset, numResults=batch_size)

            if not search_result or "models" not in search_result:
                break

            batch = search_result["models"]
            if len(batch) == 0:
                break

            models.extend([m["id"] for m in batch])

            logger.info(f"  Fetched {len(models)} models so far...")

            if len(batch) < batch_size:
                break

            offset += batch_size

        logger.info(f"Found {len(models)} total models in BioModels")

        ode_models = []
        if ode_only:
            ode_models = filter_ode_models(models)
            result_models = ode_models
        else:
            result_models = models

        registry_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "timestamp": time.time(),
            "date": datetime.now().isoformat(),
            "models": models,
        }
        if ode_only:
            cache_data["ode_models"] = ode_models

        with open(registry_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        return result_models

    except Exception as e:
        logger.error(f"Failed to query BioModels: {e}")
        return []


def select_models_to_fetch(
    n: int,
    strategy: str,
    target_total: int | None = None,
    ode_only: bool = True,
) -> list[str]:
    """Select which models to fetch.

    Automatically avoids re-fetching models already downloaded.
    """
    available = query_available_models(ode_only=ode_only)
    if not available:
        logger.error("No models available")
        return []

    already_fetched = get_already_fetched_ids()
    logger.info(f"Already have {len(already_fetched)} models")

    candidates = [m for m in available if m not in already_fetched]
    logger.info(f"{len(candidates)} candidates available")

    if not candidates:
        logger.info("All available models already fetched!")
        return []

    if target_total:
        n_needed = target_total - len(already_fetched)
        n = max(0, n_needed)
        logger.info(f"Target total: {target_total}, need {n} more")

    n = min(n, len(candidates))

    selected = random.sample(candidates, n) if strategy == "random" else candidates[:n]

    logger.info(f"Selected {len(selected)} models to fetch")
    return selected


def download_sbml(model_id: str) -> bool:
    """Download SBML file for a model.

    Returns True if successful, False otherwise.
    """
    output_path = Path(config.SBML_DOWNLOADS_DIR) / f"{model_id}.xml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return True

    try:
        import zipfile

        from bioservices import BioModels

        bm = BioModels()

        _ = bm.get_model_download(model_id)

        zip_path = Path(f"{model_id}.zip")

        if not zip_path.exists():
            logger.warning(f"Zip file not found for {model_id}")
            return False

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # BioModels uses OMEX/COMBINE archives
                sbml_files = [f for f in zf.namelist() if f.endswith("_url.xml")]

                if not sbml_files:
                    sbml_files = [
                        f for f in zf.namelist() if f.endswith(".xml") and "sbml" in f.lower()
                    ]

                if not sbml_files:
                    sbml_files = [
                        f
                        for f in zf.namelist()
                        if f.endswith(".xml") and "manifest" not in f.lower()
                    ]

                if sbml_files:
                    sbml_content = zf.read(sbml_files[0]).decode("utf-8")
                    with open(output_path, "w") as f:
                        f.write(sbml_content)
                    return True
                else:
                    logger.warning(f"No SBML file in zip for {model_id}")
                    return False
        except zipfile.BadZipFile:
            logger.error(f"Invalid zip file for {model_id}")
            return False
        finally:
            if zip_path.exists():
                zip_path.unlink()

    except Exception as e:
        logger.error(f"Failed to download {model_id}: {e}")
        zip_path = Path(f"{model_id}.zip")
        if zip_path.exists():
            zip_path.unlink()
        return False


def fetch_model(model_id: str) -> FetchResult:
    """Fetch a single model (SBML only)."""
    result: FetchResult = {
        "model_id": model_id,
        "sbml_success": False,
        "error": None,
    }

    try:
        result["sbml_success"] = download_sbml(model_id)
    except Exception as e:
        result["error"] = f"SBML download: {e}"
        return result

    if not result["sbml_success"]:
        result["error"] = "SBML download failed"

    return result


def record_fetch_session(model_ids: list[str], results: list[dict]):
    """Record this fetch session in history."""
    history = load_fetch_history()

    successful_ids = [r["model_id"] for r in results if r["sbml_success"]]

    session = {
        "session_id": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        "timestamp": datetime.now().isoformat(),
        "n_requested": len(model_ids),
        "n_fetched": len(successful_ids),
        "model_ids": successful_ids,
    }

    history["fetch_sessions"].append(session)

    all_ids = set(history.get("all_fetched_ids", []))
    all_ids.update(successful_ids)
    history["all_fetched_ids"] = sorted(all_ids)
    history["total_unique_models"] = len(all_ids)

    save_fetch_history(history)

    logger.info(f"Session complete: {len(successful_ids)}/{len(model_ids)} models fetched")
    logger.info(f"Total unique models: {history['total_unique_models']}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Fetch models from BioModels database")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of models to fetch",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help="Fetch until we have this many total models",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=config.FETCH_STRATEGIES,
        default=config.DEFAULT_STRATEGY,
        help="Selection strategy (random or sequential)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Fetch ALL models including non-ODE",
    )

    args = parser.parse_args()

    if args.n is None and args.target_total is None:
        parser.error("Must specify either --n or --target-total")

    utils.setup_logging(config.LOG_LEVEL, config.LOG_FILE)

    logger.info("=" * 60)
    logger.info("BioModels Fetch Script")
    logger.info("=" * 60)
    logger.info(f"Strategy: {args.strategy}")

    ode_only = not args.all_models
    if ode_only:
        logger.info("Filtering for ODE models only")
    else:
        logger.info("Fetching ALL models (including non-ODE)")

    models_to_fetch = select_models_to_fetch(
        n=args.n or 0,
        strategy=args.strategy,
        target_total=args.target_total,
        ode_only=ode_only,
    )

    if not models_to_fetch:
        logger.info("No models to fetch. Done!")
        return

    results = []
    for model_id in tqdm(models_to_fetch, desc="Fetching models"):
        result = fetch_model(model_id)
        results.append(result)

    record_fetch_session(models_to_fetch, results)

    failures = [r for r in results if not r["sbml_success"]]
    if failures:
        logger.warning(f"\n{len(failures)} models failed:")
        for f in failures[:10]:
            logger.warning(f"  {f['model_id']}: {f['error']}")
        if len(failures) > 10:
            logger.warning(f"  ... and {len(failures) - 10} more")

    logger.info("\nFetch complete!")


if __name__ == "__main__":
    main()
