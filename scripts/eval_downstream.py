#!/usr/bin/env python3
"""Evaluate saved detection checkpoints; save only numeric results and config."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[variable] = "1"

from config.datasets import DATASETS
from config.methods import GRAM_GRID_BETAS, GRAM_VARIANTS, resolve_gram_variant
from config.path import PATHS
from src.downstream_experiment import DownstreamScenario, run_downstream_experiment
from src.methods import METHODS
from src.utils.run_logging import run_logger
from scripts.best_k import get_best_k


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--method", required=True, choices=METHODS)
    gram_selection = parser.add_mutually_exclusive_group()
    gram_selection.add_argument(
        "--gram_variant", choices=tuple(GRAM_VARIANTS), default=None, metavar="VARIANT",
        help="explicit saved GRAM variant; otherwise choose best k at --beta (default 1)",
    )
    gram_selection.add_argument("--beta", type=float, choices=GRAM_GRID_BETAS,
                                help="select k for this dataset/noise/rho/beta using the shared API")
    parser.add_argument("--noise_type", choices=("symmetric", "pairflip", "instance"), default="instance")
    parser.add_argument("--rho", type=float, choices=(0.20, 0.40), default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoints", type=float, nargs="+", default=None,
        help="evaluate a subset of saved budget fractions; default: every saved checkpoint",
    )
    parser.add_argument("--data-root", type=Path, default=PATHS.data_root)
    parser.add_argument("--results-root", type=Path, default=PATHS.results_root)
    parser.add_argument("--overwrite", action="store_true", help="replace downstream outputs only")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_logger.setLevel(logging.INFO)
    run_logger.propagate = False
    for handler in run_logger.handlers[:]:
        handler.close()
        run_logger.removeHandler(handler)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    run_logger.addHandler(handler)
    try:
        if args.beta is not None and args.method != "ours":
            raise ValueError("--beta is only valid for --method ours")
        variant = args.gram_variant
        if args.method == "ours" and variant is None:
            choice = get_best_k(args.dataset, args.noise_type, args.rho,
                                1.0 if args.beta is None else args.beta,
                                results_root=args.results_root)
            if args.seed not in {int(seed) for seed in choice["seeds"].split(",")}:
                raise ValueError(f"selected {choice['gram_variant']} lacks completed seed{args.seed}")
            variant = choice["gram_variant"]
            run_logger.info("shared_best_k beta=%s k=%s variant=%s", choice["beta"],
                            choice["graph_neighbors"], variant)
        variant = resolve_gram_variant(args.method, variant)
        source = args.results_root / "detection" / args.dataset / args.noise_type / f"rho{args.rho:g}" / args.method
        if variant is not None:
            source /= variant
        source = source / f"seed{args.seed}" / "config.json"
        with source.open(encoding="utf-8") as stream:
            saved_config = json.load(stream)
        scenario = DownstreamScenario(
            args.noise_type, args.rho, tuple(saved_config["scenario"]["budgets"]), args.seed,
        )
        output = run_downstream_experiment(
            dataset=args.dataset, method=args.method, scenario=scenario,
            data_root=args.data_root, results_root=args.results_root,
            gram_variant=variant,
            checkpoints=None if args.checkpoints is None else tuple(args.checkpoints),
            overwrite=args.overwrite,
        )
    except Exception:
        run_logger.exception("downstream_failed; saved detection results were not modified")
        return 1
    run_logger.info("downstream_completed output=%s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
