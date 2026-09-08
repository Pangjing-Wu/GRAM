#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Keep cache reads offline; missing assets are prepared by an online subprocess.
for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[variable] = "1"

from config.datasets import DATASETS
from config.methods import GRAM_VARIANTS, resolve_gram_variant
from config.path import PATHS
from src.downstream_experiment import (
    DownstreamScenario,
    downstream_output_dir,
    run_downstream_experiment,
)
from src.methods import METHODS
from src.utils.budget import noise_rate_budget_checkpoints
from src.utils.run_logging import (
    configure_run_logging,
    ensure_output_available,
    write_run_arguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the independent full-pool downstream protocol."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument(
        "--gram_variant",
        choices=tuple(GRAM_VARIANTS),
        default=None,
        help="GRAM configuration variant; defaults to the main variant for --method=ours",
    )
    parser.add_argument(
        "--noise_type",
        choices=("symmetric", "pairflip", "instance"),
        default="instance",
        help="label corruption mechanism: uniform-other-class, cyclic pair, or feature-dependent",
    )
    parser.add_argument(
        "--rho",
        type=float,
        choices=(0.20, 0.40),
        default=0.20,
        help=(
            "fraction of D whose labels are changed; verification budgets use dense "
            "low-budget checkpoints followed by 0.05 increments through this rate"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace files in an existing result directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_gram_variant = resolve_gram_variant(args.method, args.gram_variant)
    budgets = noise_rate_budget_checkpoints(args.rho)
    scenario = DownstreamScenario(
        args.noise_type,
        args.rho,
        budgets,
        args.seed,
    )
    output = downstream_output_dir(
        PATHS.results_root,
        args.dataset,
        args.method,
        scenario,
        selected_gram_variant,
    )
    ensure_output_available(output, args.overwrite)
    write_run_arguments(
        output,
        {
            "protocol": "downstream",
            **vars(args),
            "gram_variant": selected_gram_variant,
            "budget_checkpoints": list(budgets),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "command": shlex.join(sys.argv),
            "output_dir": str(output),
        },
    )
    logger = configure_run_logging(output, overwrite=args.overwrite)
    started = time.perf_counter()
    logger.info(
        "run_started protocol=downstream dataset=%s method=%s gram_variant=%s "
        "noise=%s rho=%g seed=%d cuda_visible_devices=%s command=%s output_dir=%s",
        args.dataset,
        args.method,
        selected_gram_variant,
        args.noise_type,
        args.rho,
        args.seed,
        os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        shlex.join(sys.argv),
        output,
    )
    try:
        output = run_downstream_experiment(
            dataset=args.dataset,
            method=args.method,
            scenario=scenario,
            data_root=PATHS.data_root,
            results_root=PATHS.results_root,
            gram_variant=selected_gram_variant,
        )
    except BaseException:
        logger.exception(
            "run_failed protocol=downstream elapsed_seconds=%.3f",
            time.perf_counter() - started,
        )
        raise
    logger.info(
        "run_completed protocol=downstream elapsed_seconds=%.3f output_dir=%s",
        time.perf_counter() - started,
        output,
    )


if __name__ == "__main__":
    main()
