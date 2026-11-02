#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cli_args=("$@")
datasets=() methods=() gram_variants=() betas=() noise_types=() rhos=() seeds=() gpus=()
extra_args=()
dry_run=false
workers_per_gpu=4

while (($#)); do
  option=$1
  case "$option" in
    --datasets|--methods|--gram-variants|--betas|--noise-types|--rhos|--seeds|--gpus)
      key=${option#--}
      key=${key//-/_} ;;
    --baselines) key=methods ;;
    --overwrite) extra_args+=(--overwrite); shift; continue ;;
    --dry-run) dry_run=true; shift; continue ;;
    --workers-per-gpu)
      (($# >= 2)) && [[ $2 =~ ^[1-9][0-9]*$ ]] || { echo "--workers-per-gpu requires a positive integer" >&2; exit 2; }
      workers_per_gpu=$2
      shift 2; continue ;;
    --results-root|--data-root)
      (($# >= 2)) && [[ $2 != --* ]] || { echo "Missing value for $option" >&2; exit 2; }
      extra_args+=("$option" "$2")
      shift 2; continue ;;
    -h|--help)
      echo "Usage: $0 --gpus ... [--workers-per-gpu N] [--datasets ...] [--methods ...] [--noise-types ...] [--rhos ...] [--seeds ...] [--betas ... | --gram-variants ...] [--results-root PATH] [--data-root PATH] [--dry-run] [--overwrite]"
      echo "Default: discover existing detection scenarios, betas and seeds; automatically select the best GRAM k per dataset/noise/rho/beta. Baselines use their completed detection runs."
      echo "Use --gram-variants to override automatic k selection with explicit saved configurations."
      echo "Completed downstream runs are skipped unless --overwrite is supplied."
      echo "Default: 4 independent processes per GPU; workers share a dynamic task queue. Use --workers-per-gpu 1 for one process per GPU."
      exit 0 ;;
    *) echo "Unknown option: $option" >&2; exit 2 ;;
  esac
  shift
  if (($# == 0)) || [[ $1 == --* ]]; then
    echo "Missing values for $option" >&2; exit 2
  fi
  declare -n values="$key"
  while (($#)) && [[ $1 != --* ]]; do values+=("$1"); shift; done
done

((${#gpus[@]})) || { echo "Missing --gpus" >&2; exit 2; }
if ((${#gram_variants[@]} && ${#betas[@]})); then
  echo "Use either --betas or --gram-variants to filter saved GRAM configurations." >&2; exit 2
fi

# Materialize only completed detection runs; missing scenarios are never expanded
# into jobs. Keep the caller's working directory for relative root arguments.
plan_file=$(mktemp)
trap 'rm -f -- "$plan_file"' EXIT
PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" python3 - "${cli_args[@]}" > "$plan_file" <<'PY'
import argparse
import json
import sys

from config.datasets import DATASETS
from config.methods import GRAM_GRID_BETAS, GRAM_GRID_NEIGHBORS, GRAM_VARIANTS, METHOD_CONFIGS
from config.path import PATHS
from pathlib import Path
from scripts.best_k import get_best_k, variant_for
from src.utils.gram_selection import NOISE_TYPES, RHOS, REQUIRED_FILES
from src.utils.run_logging import completed_downstream_run_exists, completed_run_exists

parser = argparse.ArgumentParser(prog="run_downstream_experiments.sh")
for flags, choices, convert in (
    (("--datasets",), tuple(DATASETS), str),
    (("--methods", "--baselines"), tuple(METHOD_CONFIGS), str),
    (("--noise-types",), NOISE_TYPES, str),
    (("--rhos",), RHOS, float),
    (("--seeds",), None, int),
    (("--gpus",), None, str),
    (("--betas",), GRAM_GRID_BETAS, float),
    (("--gram-variants",), tuple(GRAM_VARIANTS), str),
):
    parser.add_argument(*flags, nargs="+", action="extend", choices=choices,
                        type=convert, required=flags == ("--gpus",))
parser.add_argument("--results-root", type=Path, default=PATHS.results_root)
parser.add_argument("--data-root", type=Path, default=PATHS.data_root)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--workers-per-gpu", type=int, default=4)
args = parser.parse_args()
for name in ("datasets", "methods", "noise_types", "rhos", "seeds", "gpus", "betas", "gram_variants"):
    values = getattr(args, name)
    if values and len(set(values)) != len(values):
        parser.error(f"--{name.replace('_', '-')} must not contain duplicates")
if args.seeds and any(seed < 0 for seed in args.seeds):
    parser.error("--seeds must be non-negative")

root = args.results_root / "detection"
paths = sorted([*root.glob("*/*/rho*/*/seed*/config.json"),
                *root.glob("*/*/rho*/ours/*/seed*/config.json")])
eligible = []
auto_groups = {}
full_model_betas = {variant_for(beta, k): beta
                   for beta in GRAM_GRID_BETAS for k in GRAM_GRID_NEIGHBORS}
incomplete = completed = incomplete_groups = 0
for path in paths:
    parts = path.relative_to(root).parts
    dataset, noise, rho_name, method = parts[:4]
    variant = parts[4] if len(parts) == 7 and method == "ours" else None
    if (dataset not in DATASETS or noise not in NOISE_TYPES or method not in METHOD_CONFIGS
            or (method == "ours" and variant not in GRAM_VARIANTS)):
        continue
    try:
        rho, seed = float(rho_name[3:]), int(path.parent.name[4:])
    except ValueError:
        continue
    if rho not in RHOS or seed < 0 or rho_name != f"rho{rho:g}" or path.parent.name != f"seed{seed}":
        continue
    if any(selected is not None and value not in selected for value, selected in (
        (dataset, args.datasets), (method, args.methods), (noise, args.noise_types),
        (rho, args.rhos),
    )):
        continue
    automatic = method == "ours" and not args.gram_variants
    if not automatic and args.seeds is not None and seed not in args.seeds:
        continue
    if method == "ours" and args.gram_variants and variant not in args.gram_variants:
        continue
    if automatic and (variant not in full_model_betas
                      or (args.betas and full_model_betas[variant] not in args.betas)):
        continue
    if not completed_run_exists(path.parent, protocol="full_pool_detection", required_files=REQUIRED_FILES):
        incomplete += 1
        continue
    try:
        config = json.loads(path.read_text())
        scenario = config["scenario"]
        if (config["dataset"] != dataset or config["method"] != method
                or config.get("gram_variant") != variant
                or config["protocol"] != "full_pool_detection"
                or (scenario["noise_type"], scenario["rho"], scenario["seed"]) != (noise, rho, seed)):
            raise ValueError("saved detection identity does not match its directory")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"invalid detection config {path}: {error}")
    if automatic:
        group = (dataset, noise, rho, full_model_betas[variant])
        auto_groups.setdefault(group, {}).setdefault(variant, set()).add(seed)
    else:
        eligible.append((dataset, method, variant or "-", noise, f"{rho:g}", str(seed)))

for (dataset, noise, rho, beta), candidates in sorted(auto_groups.items()):
    # Keep the shared selector's complete-five-k rule; a partially finished
    # detection sweep is not evidence for choosing among all k candidates.
    if any(0 not in candidates.get(variant_for(beta, k), set()) for k in GRAM_GRID_NEIGHBORS):
        incomplete_groups += 1
        continue
    try:
        choice = get_best_k(dataset, noise, rho, beta, results_root=args.results_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"cannot select k for {dataset}/{noise}/rho{rho:g}/beta{beta:g}: {error}")
    available = {int(seed) for seed in choice["seeds"].split(",")}
    if args.seeds is not None and not set(args.seeds) <= available:
        parser.error(f"selected {dataset}/{noise}/rho{rho:g}/beta{beta:g}: "
                     f"k={choice['graph_neighbors']} ({choice['gram_variant']}) only has seeds "
                     f"{sorted(available)}; requested {args.seeds}")
    for seed in sorted(available if args.seeds is None else args.seeds):
        eligible.append((dataset, "ours", choice["gram_variant"], noise, f"{rho:g}", str(seed)))
    print(f"Selected {dataset}/{noise}/rho{rho:g} beta={beta:g} "
          f"k={choice['graph_neighbors']} variant={choice['gram_variant']} "
          f"(seeds={choice['seeds']})", file=sys.stderr)

jobs = []
for job in eligible:
    dataset, method, variant, noise, rho, seed = job
    output = args.results_root / "downstream" / dataset / noise / f"rho{rho}" / method
    if variant != "-":
        output /= variant
    output /= f"seed{seed}"
    if not args.overwrite and completed_downstream_run_exists(output):
        completed += 1
        continue
    jobs.append(job)

print(f"Planned {len(jobs)} downstream runs; skipped {incomplete} incomplete detection runs, "
      f"{incomplete_groups} incomplete k-selection groups "
      f"and {completed} completed downstream runs.", file=sys.stderr)
for job in jobs:
    print("\t".join(job))
PY

python3 - "$project_root" "$plan_file" "$workers_per_gpu" "$dry_run" \
  "${#gpus[@]}" "${gpus[@]}" "${extra_args[@]}" <<'PY'
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from queue import Empty, Queue
import shlex
import signal
import subprocess
import sys
import time
from threading import Event, Lock

project_root, plan_file, workers_per_gpu, dry_run, gpu_count, *remaining = sys.argv[1:]
gpu_count, workers_per_gpu = int(gpu_count), int(workers_per_gpu)
gpus, extra_args = remaining[:gpu_count], remaining[gpu_count:]
jobs = [line.split("\t") for line in Path(plan_file).read_text().splitlines()]
print(f"Concurrency: {workers_per_gpu} processes/GPU × {gpu_count} GPUs "
      f"= {workers_per_gpu * gpu_count} slots, shared task queue.", file=sys.stderr)

def command_for(job):
    dataset, method, variant, noise, rho, seed = job
    command = ["conda", "run", "--no-capture-output", "-n", "torch231", "python",
               str(Path(project_root) / "scripts/eval_downstream.py"),
               "--dataset", dataset, "--method", method,
               "--noise_type", noise, "--rho", rho, "--seed", seed]
    if variant != "-":
        command += ["--gram_variant", variant]
    return [*command, *extra_args]

if dry_run == "true":
    for index, job in enumerate(jobs):
        print(f"CUDA_VISIBLE_DEVICES={shlex.quote(gpus[index % gpu_count])} "
              f"{shlex.join(command_for(job))}")
    sys.exit(0)
if not jobs:
    sys.exit(0)

queue = Queue()
for index, job in enumerate(jobs, 1):
    queue.put((index, job))
stopping, lock = Event(), Lock()
active = set()

def worker(gpu):
    failures = 0
    while not stopping.is_set():
        try:
            index, job = queue.get_nowait()
        except Empty:
            break
        try:
            with lock:
                if stopping.is_set():
                    break
                print(f"START [{index}/{len(jobs)}] GPU={gpu} {'/'.join(job)}", file=sys.stderr, flush=True)
                process = subprocess.Popen(command_for(job),
                                           env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                           start_new_session=True)
                active.add(process)
            code = process.wait()
            with lock:
                active.discard(process)
            if code:
                failures += 1
                print(f"FAILED [{index}/{len(jobs)}] GPU={gpu} exit={code} {'/'.join(job)}",
                      file=sys.stderr, flush=True)
        except OSError as error:
            failures += 1
            print(f"FAILED [{index}/{len(jobs)}] GPU={gpu}: {error}", file=sys.stderr, flush=True)
    return failures

interrupted_signal = signal.SIGINT

def interrupt(signum, frame):
    global interrupted_signal
    interrupted_signal = signum
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, interrupt)
signal.signal(signal.SIGTERM, interrupt)
slots = [gpu for _ in range(workers_per_gpu) for gpu in gpus]
pool = ThreadPoolExecutor(max_workers=min(len(slots), len(jobs)))
status = 0
try:
    futures = [pool.submit(worker, gpu) for gpu in slots[:len(jobs)]]
    failures = sum(future.result() for future in futures)
    print(f"Finished {len(jobs)} downstream tasks; {failures} failed.", file=sys.stderr)
    status = int(failures > 0)
except KeyboardInterrupt:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    stopping.set()
    with lock:
        processes = list(active)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for process in processes:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            deadline = time.monotonic() + 3
            for process in processes:
                try:
                    process.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
    status = 128 + interrupted_signal
finally:
    pool.shutdown(wait=True)
sys.exit(status)
PY
