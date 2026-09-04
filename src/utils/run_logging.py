from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


LOGGER_NAME = "data_model_coevo.run"
run_logger = logging.getLogger(LOGGER_NAME)


def completed_run_exists(
    output_dir: str | Path,
    *,
    protocol: str,
    required_files: tuple[str, ...] = (),
) -> bool:
    """Return whether an output directory contains a completed run."""
    output_dir = Path(output_dir)
    if any(not (output_dir / name).is_file() for name in required_files):
        return False

    log_path = output_dir / "run.log"
    if not log_path.is_file():
        return False

    completion_marker = f"run_completed protocol={protocol}"
    try:
        with log_path.open(encoding="utf-8") as stream:
            return any(completion_marker in line for line in stream)
    except OSError:
        return False


def ensure_output_available(output_dir: str | Path, overwrite: bool = False) -> None:
    """Refuse to mix a new run with files from an existing result."""
    output_dir = Path(output_dir)
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"result directory is not empty: {output_dir}; "
            "pass --overwrite to replace result files"
        )


def write_run_arguments(
    output_dir: str | Path, arguments: Mapping[str, Any]
) -> Path:
    """Persist the effective invocation parameters before a run starts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_args.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(dict(arguments), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def configure_run_logging(
    output_dir: str | Path, *, overwrite: bool = False
) -> logging.Logger:
    """Write one run's lifecycle and progress to its result directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s pid=%(process)d %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime

    file_handler = logging.FileHandler(
        output_dir / "run.log", mode="w" if overwrite else "a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
