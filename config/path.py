from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    data_root: Path
    results_root: Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AMADE_ROOT = PROJECT_ROOT.parent / "amade"

PATHS = PathConfig(
    data_root=AMADE_ROOT / "datasets",
    results_root=PROJECT_ROOT / "results",
)
