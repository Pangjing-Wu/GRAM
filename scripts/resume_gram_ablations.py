#!/usr/bin/env python3
"""Resume unfinished runs in the six-dataset detection ablation grid."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.methods import GRAM_ABLATION_VARIANTS
from config.path import PATHS
from src.utils.run_logging import completed_run_exists

DATASETS = ("cifar10", "cifar100", "atis", "qnli", "adult", "letter")
REQUIRED_FILES = (
    "run_args.json", "metrics.csv", "queries.csv", "predictions.csv",
    "summary.json", "config.json",
)


def completed(output: Path) -> bool:
    return completed_run_exists(
        output, protocol="full_pool_detection", required_files=REQUIRED_FILES,
    )


def missing_runs():
    for dataset, variant, noise, rho, seed in product(
        DATASETS, GRAM_ABLATION_VARIANTS,
        ("symmetric", "pairflip", "instance"), ("0.2", "0.4"), range(3),
    ):
        output = (
            PATHS.results_root / "detection" / dataset / noise / f"rho{rho}"
            / "ours" / variant / f"seed{seed}"
        )
        if not completed(output):
            yield dataset, variant, noise, rho, seed, output


def run_worker(gpu: str, queue: Queue, conda_env: str, retries: int):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
    failures = []
    while True:
        try:
            dataset, variant, noise, rho, seed, output = queue.get_nowait()
        except Empty:
            return failures
        # Recheck in case a run finished after the initial inventory.
        if completed(output):
            continue
        label = f"{dataset}/{noise}/rho{rho}/{variant}/seed{seed}"
        command = [
            "conda", "run", "--no-capture-output", "-n", conda_env,
            "python", str(PROJECT_ROOT / "scripts" / "eval_detection.py"),
            "--dataset", dataset, "--method", "ours", "--gram_variant", variant,
            "--noise_type", noise, "--rho", rho, "--seed", str(seed),
        ]
        for attempt in range(retries + 1):
            if completed(output):
                break
            # Only incomplete runs may have their partial output replaced.
            overwrite = ["--overwrite"] if output.is_dir() and any(output.iterdir()) else []
            print(f"[GPU {gpu}] attempt {attempt + 1}/{retries + 1}: {label}", flush=True)
            result = subprocess.run(command + overwrite, cwd=PROJECT_ROOT, env=env)
            if result.returncode == 0 and completed(output):
                print(f"[GPU {gpu}] completed: {label}", flush=True)
                break
            print(f"[GPU {gpu}] failed (exit {result.returncode}): {label}", flush=True)
        else:
            failures.append(label)
        # A failed run does not stop this GPU's remaining jobs.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--conda-env", default="torch231")
    parser.add_argument("--retries", type=int, default=1,
                        help="extra attempts per failed run (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list missing runs without executing or changing results")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU IDs must be unique")
    if args.retries < 0:
        parser.error("--retries must be nonnegative")

    pending = list(missing_runs())
    print(f"Unfinished ablations: {len(pending)}", flush=True)
    counts = Counter(run[0] for run in pending)
    for dataset in DATASETS:
        print(f"  {dataset}: {counts[dataset]}", flush=True)
    if args.dry_run:
        for dataset, variant, noise, rho, seed, _ in pending:
            print(f"{dataset}\t{variant}\t{noise}\t{rho}\t{seed}")
        return 0
    if not pending:
        return 0
    if shutil.which("conda") is None:
        parser.error("conda executable not found in PATH")

    queue = Queue()
    for run in pending:
        queue.put(run)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(run_worker, gpu, queue, args.conda_env, args.retries)
                   for gpu in args.gpus]
        failures = [label for future in futures for label in future.result()]
    print(f"Finished. Failed runs: {len(failures)}", flush=True)
    for label in failures:
        print(f"FAILED\t{label}", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
