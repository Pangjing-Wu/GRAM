#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
echo "compatibility entry point: running the detection protocol" >&2
echo "use run_detection_experiments.sh or run_downstream_experiments.sh explicitly" >&2
exec bash "$project_root/scripts/run_detection_experiments.sh" "$@"
