#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
suite=${1:-pilot}
gpu_csv=${2:-0,1,2,3}
IFS=',' read -r -a gpus <<< "$gpu_csv"

methods=(aum_b el2n_b knn_label_disagreement_b moderate_b robust_alc dalc active_label_cleaning active_label_correction graph_label_propagation cleannet misdetect_b ours)
seeds=(0)
commands=()

add_grid() {
  local rho=$1
  shift
  local datasets=("$@")
  local dataset method noise seed
  for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
      if [[ $method == cleannet && $dataset != cifar10 && $dataset != cifar100 ]]; then
        continue
      fi
      for noise in symmetric pairflip instance; do
        for seed in "${seeds[@]}"; do
          commands+=("$dataset|$method|$noise|$rho|$seed")
        done
      done
    done
  done
}

case "$suite" in
  pilot)
    for dataset in cifar10 atis adult; do
      for method in "${methods[@]}"; do
        if [[ $method == cleannet && $dataset != cifar10 && $dataset != cifar100 ]]; then
          continue
        fi
        for rho in 0.2 0.4; do
          commands+=("$dataset|$method|symmetric|$rho|0")
        done
      done
    done
    ;;
  main)
    for rho in 0.2 0.4; do
      add_grid "$rho" cifar10 cifar100 atis qnli adult letter
    done
    ;;
  *)
    echo "usage: $0 {pilot|main} [gpu_ids]" >&2
    exit 2
    ;;
esac

run_worker() {
  local gpu=$1
  local offset=$2
  local spec dataset method noise rho seed
  for ((index=offset; index<${#commands[@]}; index+=${#gpus[@]})); do
    spec=${commands[index]}
    IFS='|' read -r dataset method noise rho seed <<< "$spec"
    CUDA_VISIBLE_DEVICES=$gpu conda run -n torch231 python \
      "$project_root/scripts/eval_downstream.py" \
      --dataset "$dataset" \
      --method "$method" \
      --noise_type "$noise" \
      --rho "$rho" \
      --seed "$seed"
  done
}

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
