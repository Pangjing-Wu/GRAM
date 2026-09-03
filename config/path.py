from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    data_root: Path
    results_root: Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATHS = PathConfig(
    data_root=PROJECT_ROOT / "datasets",
    results_root=PROJECT_ROOT / "results",
)
