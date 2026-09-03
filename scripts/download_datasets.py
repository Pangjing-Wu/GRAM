#!/usr/bin/env python3
"""Download the benchmark datasets into the project's local data directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.datasets import DATASETS
from config.path import PATHS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets before running experiments."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
        help="datasets to download (default: all configured datasets)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PATHS.data_root,
        help=f"download directory (default: {PATHS.data_root})",
    )
    return parser.parse_args()


def download_cifar(config: dict, root: Path) -> None:
    from datasets import load_dataset

    load_dataset(
        "parquet",
        data_files=config["data_files"],
        cache_dir=str(root / "huggingface"),
    )


def download_huggingface(config: dict, root: Path) -> None:
    from datasets import load_dataset

    arguments = [config["source"]]
    if "subset" in config:
        arguments.append(config["subset"])
    load_dataset(*arguments, cache_dir=str(root / "huggingface"))


def download_openml(config: dict, root: Path) -> None:
    from sklearn.datasets import fetch_openml

    _, name, version = config["source"].split(":", maxsplit=2)
    fetch_openml(
        name=name,
        version=int(version),
        data_home=str(root / "openml"),
    )


def download_dataset(name: str, root: Path) -> None:
    config = DATASETS[name]
    source = config["source"]
    if name in {"cifar10", "cifar100"}:
        download_cifar(config, root)
    elif source.startswith("openml:"):
        download_openml(config, root)
    else:
        download_huggingface(config, root)


def main() -> None:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"Data root: {root}", flush=True)

    failures: list[tuple[str, Exception]] = []
    for name in args.datasets:
        print(f"[{name}] downloading...", flush=True)
        try:
            download_dataset(name, root)
        except Exception as error:  # Continue so independent downloads can finish.
            failures.append((name, error))
            print(f"[{name}] failed: {error}", file=sys.stderr, flush=True)
        else:
            print(f"[{name}] ready", flush=True)

    if failures:
        failed_names = ", ".join(name for name, _ in failures)
        raise SystemExit(f"Failed datasets: {failed_names}")
    print("All requested datasets are ready.", flush=True)


if __name__ == "__main__":
    main()
