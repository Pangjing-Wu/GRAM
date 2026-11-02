#!/usr/bin/env bash
set -euo pipefail

# bash scripts/run_detection_experiments.sh --datasets cifar10 cifar100 atis qnli adult letter --methods aum_b el2n_b forgetting_b early_loss_b cleanlab_b knn_label_disagreement_b moderate_b robust_alc robust_alc_frozen dalc dalc_frozen active_label_cleaning active_label_correction graph_label_propagation cleannet misdetect_b ours --noise-types symmetric pairflip instance --rhos 0.2 0.4 --neighbors 10 25 50 100 250 --betas 0 0.1 1 10 100 1000 10000 --seeds 0 1 2 --gpus 0 1

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cli_args=("$@")
datasets=() methods=() gram_variants=() betas=() neighbors=() noise_types=() rhos=() seeds=() gpus=()
extra_args=()
dry_run=false

while (($#)); do
  option=$1
  case "$option" in
    --datasets|--methods|--gram-variants|--betas|--neighbors|--noise-types|--rhos|--seeds|--gpus)
      key=${option#--}
      key=${key//-/_} ;;
    --baselines) key=methods ;;
    --overwrite) extra_args=(--overwrite); shift; continue ;;
    --dry-run) dry_run=true; shift; continue ;;
    -h|--help)
      echo "Usage: $0 --datasets ... --methods ... --noise-types ... --rhos ... --seeds ... --gpus ... [--betas ...] [--neighbors ...] [--gram-variants ...] [--dry-run] [--overwrite]"
      echo "GRAM defaults to beta=0/0.1/1/10/100/1000/10000 crossed with k=10/25/50/100/250, using adaptive kernels and repair+IV (risk-only at beta=0)."
      echo "Use --gram-variants for explicit configurations instead of --betas/--neighbors. Baselines run once per scenario/seed."
      exit 0 ;;
    *) echo "Unknown option: $option" >&2; exit 2 ;;
  esac
  shift
  if (($# == 0)) || [[ $1 == --* ]]; then
    echo "Missing values for $option" >&2
    exit 2
  fi
  declare -n values="$key"
  while (($#)) && [[ $1 != --* ]]; do
    values+=("$1")
    shift
  done
done

for key in datasets methods noise_types rhos seeds gpus; do
  declare -n values="$key"
  if ((${#values[@]} == 0)); then
    echo "Missing --${key//_/-}" >&2
    exit 2
  fi
done
if ((${#gram_variants[@]} && (${#betas[@]} || ${#neighbors[@]}))); then
  echo "Use either --betas/--neighbors or --gram-variants." >&2
  exit 2
fi

# Validate the complete request before starting workers and reuse canonical
# variant names so existing detection results remain discoverable.
variant_config=$(
  cd "$project_root"
  python3 - "${cli_args[@]}" <<'PY'
import argparse

from config.datasets import DATASETS
from config.methods import (
    GRAM_GRID_BETAS, GRAM_GRID_NEIGHBORS, GRAM_VARIANTS, METHOD_CONFIGS,
    gram_grid_variants,
)

parser = argparse.ArgumentParser(prog="run_detection_experiments.sh")
parser.add_argument("--datasets", nargs="+", action="extend", choices=tuple(DATASETS), required=True)
parser.add_argument("--methods", "--baselines", nargs="+", action="extend", choices=tuple(METHOD_CONFIGS), required=True)
parser.add_argument("--noise-types", nargs="+", action="extend", choices=("symmetric", "pairflip", "instance"), required=True)
parser.add_argument("--rhos", nargs="+", action="extend", type=float, choices=(0.2, 0.4), required=True)
parser.add_argument("--seeds", nargs="+", action="extend", type=int, required=True)
parser.add_argument("--gpus", nargs="+", action="extend", required=True)
parser.add_argument("--betas", nargs="+", action="extend", type=float, choices=GRAM_GRID_BETAS)
parser.add_argument("--neighbors", nargs="+", action="extend", type=int, choices=GRAM_GRID_NEIGHBORS)
parser.add_argument("--gram-variants", nargs="+", action="extend", choices=tuple(GRAM_VARIANTS))
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
if any(seed < 0 for seed in args.seeds):
    parser.error("--seeds must be non-negative")
for name in ("datasets", "methods", "noise_types", "rhos", "seeds", "gpus", "betas", "neighbors", "gram_variants"):
    values = getattr(args, name)
    if values and len(set(values)) != len(values):
        parser.error(f"--{name.replace('_', '-')} must not contain duplicates")
variants = args.gram_variants or gram_grid_variants(
    betas=args.betas or GRAM_GRID_BETAS,
    neighbors=args.neighbors or GRAM_GRID_NEIGHBORS,
    kernels=("adaptive",), acquisitions=("repair_iv",),
)
print("\n".join(variants))
PY
)
mapfile -t gram_variants <<< "$variant_config"

run_worker() {
  local slot=$1 index=0 dataset method variant noise rho seed
  local -a variants args
  for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
      variants=("")
      [[ $method != ours ]] || variants=("${gram_variants[@]}")
      for variant in "${variants[@]}"; do
        for noise in "${noise_types[@]}"; do
          for rho in "${rhos[@]}"; do
            for seed in "${seeds[@]}"; do
              if ((index % ${#gpus[@]} == slot)); then
                args=(--dataset "$dataset" --method "$method" --noise_type "$noise" --rho "$rho" --seed "$seed")
                [[ -z $variant ]] || args+=(--gram_variant "$variant")
                if $dry_run; then
                  printf 'CUDA_VISIBLE_DEVICES=%q conda run -n torch231 python %q ' "${gpus[slot]}" "$project_root/scripts/eval_detection.py"
                  printf '%q ' "${args[@]}" "${extra_args[@]}"
                  printf '\n'
                else
                  CUDA_VISIBLE_DEVICES="${gpus[slot]}" conda run -n torch231 python \
                    "$project_root/scripts/eval_detection.py" "${args[@]}" "${extra_args[@]}"
                fi
              fi
              index=$((index + 1))
            done
          done
        done
      done
    done
  done
}

pids=()
for slot in "${!gpus[@]}"; do
  if $dry_run; then
    run_worker "$slot"
  else
    run_worker "$slot" &
    pids+=("$!")
  fi
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
