#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

protocol=detection
variants=()
forwarded_args=()

usage() {
  cat <<EOF
usage: $0 [--protocol detection|downstream|both] [--variants VARIANT...] [batch options]

Run only GRAM ablations. By default, every registered non-main GRAM variant is
evaluated under the detection protocol. All remaining options are forwarded to
the corresponding batch runner; --datasets, --noise-types, --rhos, --seeds,
and --gpus are required there.

Examples:
  $0 --datasets cifar10 atis adult --noise-types symmetric --rhos 0.2 --seeds 0 --gpus 0
  $0 --protocol both --variants abl_no_identity abl_margin_only --datasets cifar10 --noise-types symmetric pairflip --rhos 0.2 --seeds 0 1 2 --gpus 0 1

Do not pass --methods, --baselines, or --gram-variants: this launcher fixes the
method to ours and manages the GRAM variants itself.
EOF
}

die() {
  echo "error: $*" >&2
  echo >&2
  usage >&2
  exit 2
}

while (($# > 0)); do
  case "$1" in
    --protocol)
      (($# >= 2)) || die "--protocol requires a value"
      protocol=$2
      shift 2
      ;;
    --variants)
      shift
      while (($# > 0)) && [[ $1 != --* ]]; do
        variants+=("$1")
        shift
      done
      ((${#variants[@]} > 0)) || die "--variants requires at least one value"
      ;;
    --methods|--baselines|--gram-variants)
      die "$1 is managed by this ablation launcher"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done

case "$protocol" in
  detection|downstream|both) ;;
  *) die "unsupported protocol: $protocol (expected detection, downstream, or both)" ;;
esac

if ((${#variants[@]} == 0)); then
  mapfile -t variants < <(
    cd "$project_root"
    python3 - <<'PY'
from config.methods import GRAM_ABLATION_VARIANTS

print("\n".join(GRAM_ABLATION_VARIANTS))
PY
  )
fi

run_protocol() {
  local selected_protocol=$1
  local runner="$project_root/scripts/run_${selected_protocol}_experiments.sh"
  bash "$runner" \
    --methods ours \
    --gram-variants "${variants[@]}" \
    "${forwarded_args[@]}"
}

if [[ $protocol == both ]]; then
  run_protocol detection
  run_protocol downstream
else
  run_protocol "$protocol"
fi
