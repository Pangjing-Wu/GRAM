#!/usr/bin/env bash
set -euo pipefail

# Quick validation (image, text, tabular):
# bash scripts/run_detection_experiments.sh --datasets cifar10 atis adult --methods ours aum_b el2n_b forgetting_b early_loss_b cleanlab_b knn_label_disagreement_b moderate_b robust_alc robust_alc_frozen dalc dalc_frozen active_label_cleaning active_label_correction graph_label_propagation cleannet misdetect_b --noise-types symmetric pairflip instance --rhos 0.2 --seeds 0 --gpus 0 1
# Full grid (all valid combinations):
# bash scripts/run_detection_experiments.sh --datasets cifar10 cifar100 atis qnli adult letter --methods aum_b el2n_b forgetting_b early_loss_b cleanlab_b knn_label_disagreement_b moderate_b robust_alc robust_alc_frozen dalc dalc_frozen active_label_cleaning active_label_correction graph_label_propagation cleannet misdetect_b ours --noise-types symmetric pairflip instance --rhos 0.2 0.4 --seeds 0 1 2 --gpus 0 1 2 3
# GRAM ablations: use scripts/run_gram_ablations.sh instead.

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

datasets=()
methods=()
gram_variants=()
noise_types=()
rhos=()
seeds=()
gpus=()
commands=()
overwrite=false

usage() {
  cat <<EOF
usage: $0 [options]

  --datasets DATASET...
  --methods METHOD...
  --gram-variants VARIANT...  (advanced; omitted means the main GRAM setting)
  --noise-types NOISE_TYPE...
  --rhos RHO...
  --seeds SEED...
  --gpus GPU_ID...
  --overwrite

Each option accepts one or more space-separated values. The script runs the
Cartesian product of datasets, methods, noise types, rhos, and seeds, distributing
the resulting jobs across the requested GPUs.

--baselines is accepted as an alias for --methods.
EOF
}

die() {
  echo "error: $*" >&2
  echo >&2
  usage >&2
  exit 2
}

collected_count=0
collect_values() {
  local option=$1
  local target_name=$2
  shift 2
  local -n target=$target_name

  collected_count=0
  while (($# > 0)) && [[ $1 != --* ]]; do
    target+=("$1")
    collected_count=$((collected_count + 1))
    shift
  done
  ((collected_count > 0)) || die "$option requires at least one value"
}

while (($# > 0)); do
  option=$1
  shift
  case "$option" in
    --datasets)
      collect_values "$option" datasets "$@"
      ;;
    --methods|--baselines)
      collect_values "$option" methods "$@"
      ;;
    --gram-variants)
      collect_values "$option" gram_variants "$@"
      ;;
    --noise-types)
      collect_values "$option" noise_types "$@"
      ;;
    --rhos)
      collect_values "$option" rhos "$@"
      ;;
    --seeds)
      collect_values "$option" seeds "$@"
      ;;
    --gpus)
      collect_values "$option" gpus "$@"
      ;;
    --overwrite)
      overwrite=true
      collected_count=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $option"
      ;;
  esac
  shift "$collected_count"
done

((${#datasets[@]} > 0)) || die "--datasets is required"
((${#methods[@]} > 0)) || die "--methods is required"
((${#noise_types[@]} > 0)) || die "--noise-types is required"
((${#rhos[@]} > 0)) || die "--rhos is required"
((${#seeds[@]} > 0)) || die "--seeds is required"
((${#gpus[@]} > 0)) || die "--gpus is required"

gram_variant_config=$(
  cd "$project_root"
  python3 - <<'PY'
from config.methods import DEFAULT_GRAM_VARIANT, GRAM_VARIANTS

print(DEFAULT_GRAM_VARIANT)
print("\n".join(GRAM_VARIANTS))
PY
)
mapfile -t configured_gram_variants <<< "$gram_variant_config"
default_gram_variant=${configured_gram_variants[0]}
declare -A supported_gram_variants=()
for gram_variant in "${configured_gram_variants[@]:1}"; do
  supported_gram_variants["$gram_variant"]=1
done

if ((${#gram_variants[@]} == 0)); then
  gram_variants=("$default_gram_variant")
fi

for dataset in "${datasets[@]}"; do
  case "$dataset" in
    cifar10|cifar100|atis|qnli|adult|letter) ;;
    *) die "unsupported dataset: $dataset" ;;
  esac
done

for gram_variant in "${gram_variants[@]}"; do
  [[ -n ${supported_gram_variants[$gram_variant]+x} ]] \
    || die "unsupported GRAM variant: $gram_variant"
done

for method in "${methods[@]}"; do
  case "$method" in
    aum_b|el2n_b|forgetting_b|early_loss_b|cleanlab_b|knn_label_disagreement_b|moderate_b|robust_alc|robust_alc_frozen|dalc|dalc_frozen|active_label_cleaning|active_label_correction|graph_label_propagation|cleannet|misdetect_b|ours) ;;
    *) die "unsupported method: $method" ;;
  esac
done

for noise_type in "${noise_types[@]}"; do
  case "$noise_type" in
    symmetric|pairflip|instance) ;;
    *) die "unsupported noise type: $noise_type" ;;
  esac
done

for rho in "${rhos[@]}"; do
  case "$rho" in
    0.2|0.20|0.4|0.40) ;;
    *) die "unsupported rho: $rho (expected 0.2 or 0.4)" ;;
  esac
done

for seed in "${seeds[@]}"; do
  [[ $seed =~ ^-?[0-9]+$ ]] || die "seed must be an integer: $seed"
done

for dataset in "${datasets[@]}"; do
  for method in "${methods[@]}"; do
    if [[ $method == ours ]]; then
      method_gram_variants=("${gram_variants[@]}")
    else
      method_gram_variants=("")
    fi
    for gram_variant in "${method_gram_variants[@]}"; do
      for noise_type in "${noise_types[@]}"; do
        for rho in "${rhos[@]}"; do
          for seed in "${seeds[@]}"; do
            commands+=("$dataset|$method|$gram_variant|$noise_type|$rho|$seed")
          done
        done
      done
    done
  done
done

run_worker() {
  local gpu=$1
  local offset=$2
  local spec dataset method gram_variant noise_type rho seed
  for ((index=offset; index<${#commands[@]}; index+=${#gpus[@]})); do
    spec=${commands[index]}
    IFS='|' read -r dataset method gram_variant noise_type rho seed <<< "$spec"
    command=(conda run -n torch231 python "$project_root/scripts/eval_detection.py" \
      --dataset "$dataset" \
      --method "$method" \
      --noise_type "$noise_type" \
      --rho "$rho" \
      --seed "$seed")
    if [[ -n $gram_variant ]]; then
      command+=(--gram_variant "$gram_variant")
    fi
    if [[ $overwrite == true ]]; then
      command+=(--overwrite)
    fi
    CUDA_VISIBLE_DEVICES=$gpu "${command[@]}"
  done
}

echo "Running ${#commands[@]} experiment(s) across ${#gpus[@]} GPU(s)."

pids=()
for index in "${!gpus[@]}"; do
  run_worker "${gpus[index]}" "$index" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
