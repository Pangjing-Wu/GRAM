# Label Verification Benchmark

This project provides two independent experiment protocols:

- `detection` evaluates label-issue rankings on the full training pool and the
  currently unverified subset using AUPRC and AUROC.
- `downstream` repairs or filters training data based on method predictions and
  evaluates a downstream model on the clean test set.

The protocols do not share query trajectories, model states, or result files.

## Environment

```bash
conda activate torch231
pip install -r requirements.txt
```

Configure the data and result roots in `config/path.py`.

## Run One Detection Experiment

```bash
python scripts/eval_detection.py \
  --dataset cifar10 \
  --method ours \
  --noise_type symmetric \
  --rho 0.2 \
  --seed 0
```

Results are stored under:

```text
results/detection/<dataset>/<noise_type>/<scenario>/<method>/seed<seed>/
```

The main output files are `metrics.csv`, `predictions.csv`, `queries.csv`,
`summary.json`, and `config.json`.

## Run One Downstream Experiment

```bash
python scripts/eval_downstream.py \
  --dataset cifar10 \
  --method ours \
  --noise_type symmetric \
  --rho 0.2 \
  --seed 0
```

Results are stored under:

```text
results/downstream/<dataset>/<noise_type>/rho<rho>/<method>/seed<seed>/
```

The main output files are `downstream_metrics.csv`,
`extrapolation_predictions.csv`, `queries.csv`, `summary.json`, and
`config.json`.

## Main Arguments

- `--dataset`: `cifar10`, `cifar100`, `atis`, `qnli`, `adult`, or `letter`.
- `--method`: `aum_b`, `el2n_b`, `knn_label_disagreement_b`, `moderate_b`,
  `robust_alc`, `dalc`, `active_label_cleaning`, `active_label_correction`,
  `graph_label_propagation`, `cleannet`, `misdetect_b`, or `ours`.
- `--noise_type`: `symmetric`, `pairflip`, or `instance`.
- `--rho`: label-noise rate, either `0.2` or `0.4`.
- `--seed`: random seed.

Verification checkpoints are generated automatically from `rho`: `0.001`,
`0.005`, `0.01`, and `0.025`, followed by increments of `0.05` up to `rho`.

## Batch Execution

```bash
bash scripts/run_detection_experiments.sh pilot 0,1,2,3
bash scripts/run_downstream_experiments.sh pilot 0,1,2,3
```

The second argument is a comma-separated list of GPU IDs. Replace `pilot` with
`main` to run the complete experiment grid.
