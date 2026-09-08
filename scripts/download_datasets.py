#!/usr/bin/env python3
"""Download the benchmark datasets into the project's local data directory."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.datasets import DATASETS
from config.models import MODELS
from config.path import PATHS
from src.utils.local_data import load_local_huggingface, local_openml_path


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
        # This path is entered only when the prepared Arrow cache is incomplete.
        download_mode="reuse_cache_if_exists",
    )


def download_huggingface(config: dict, root: Path) -> None:
    from datasets import load_dataset

    arguments = [config["source"]]
    if "subset" in config:
        arguments.append(config["subset"])
    load_dataset(
        *arguments, cache_dir=str(root / "huggingface"),
        download_mode="reuse_cache_if_exists",
    )


def download_openml(config: dict, root: Path) -> None:
    import joblib
    from sklearn.datasets import fetch_openml

    output = local_openml_path(root, config)
    if output.is_file():
        return
    _, name, version = config["source"].split(":", maxsplit=2)
    bunch = fetch_openml(
        name=name,
        version=int(version),
        data_home=str(root / "openml"),
        as_frame=name == "adult",
    )
    # Preserve Adult's categorical dtypes and row order for identical splits.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, suffix=".joblib", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        joblib.dump(bunch, temporary, compress=3)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def download_text_assets(name: str, root: Path) -> None:
    """Prepare the tokenizer and pretrained weights used by text experiments."""
    from transformers import AutoModel, AutoTokenizer

    model = MODELS[name]["model"]
    for loader in (AutoTokenizer, AutoModel):
        kwargs = {"cache_dir": str(root / "huggingface")}
        if loader is AutoModel:
            kwargs["add_pooling_layer"] = False
        try:
            asset = loader.from_pretrained(model, local_files_only=True, **kwargs)
        except OSError:
            asset = loader.from_pretrained(model, **kwargs)
        del asset


def download_dataset(name: str, root: Path) -> None:
    config = DATASETS[name]
    source = config["source"]
    if source.startswith("openml:"):
        download_openml(config, root)
    else:
        try:
            load_local_huggingface(name, root, config)
        except FileNotFoundError:
            if name in {"cifar10", "cifar100"}:
                download_cifar(config, root)
            else:
                download_huggingface(config, root)
            load_local_huggingface(name, root, config)
    if config["modality"] == "text":
        download_text_assets(name, root)


def main() -> None:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"Data root: {root}", flush=True)

    failures: list[tuple[str, Exception]] = []
    for name in args.datasets:
        print(f"[{name}] preparing local data...", flush=True)
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
