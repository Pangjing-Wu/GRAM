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

Datasets are stored under the project-local `datasets/` directory. To prepare
all configured datasets before running experiments, use:

```bash
python scripts/download_datasets.py
```

Use `--datasets cifar10 atis` to download only selected datasets. Data and result
roots are configured in `config/path.py`.

Experiments read local data first. CIFAR, ATIS, and QNLI are read directly from
the Arrow files under `datasets/huggingface/`, using their source metadata to
identify the correct cache. Adult and Letter are read from
`datasets/openml/<name>-v<version>.joblib`, which the download script prepares
from the OpenML cache. Rerunning the download script reuses existing local data;
it also prepares the tokenizer and pretrained weights for text experiments.
When a required dataset, tokenizer, or model weight file is missing, the
experiment automatically runs `download_datasets.py` with the same Python
environment and data root, then resumes reading local files. Only that
preparation subprocess enables network access; cache hits do not query the Hub
for remote metadata. Existing OpenML caches are converted automatically when
their prepared files are missing. Download errors propagate to the experiment.

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
results/detection/<dataset>/<noise_type>/rho<rho>/<method>/seed<seed>/
```

GRAM results include a configuration-variant level:

```text
results/detection/<dataset>/<noise_type>/rho<rho>/ours/<gram_variant>/seed<seed>/
```

The main output files are `metrics.csv`, `predictions.csv`, `queries.csv`,
`summary.json`, and `config.json`. GRAM runs additionally write
`kernel_weights.csv` with the weights and optimizer status at every checkpoint.

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

GRAM uses the corresponding
`.../ours/<gram_variant>/seed<seed>/` layout for this protocol as well.

The main output files are `downstream_metrics.csv`,
`extrapolation_predictions.csv`, `queries.csv`, `summary.json`, and
`config.json`. GRAM runs additionally write `kernel_weights.csv`.

## Main Arguments

- `--dataset`: `cifar10`, `cifar100`, `atis`, `qnli`, `adult`, or `letter`.
- `--method`: `aum_b`, `el2n_b`, `forgetting_b`, `early_loss_b`, `cleanlab_b`,
  `knn_label_disagreement_b`, `moderate_b`,
  `robust_alc`, `robust_alc_frozen`, `dalc`, `dalc_frozen`,
  `active_label_cleaning`, `active_label_correction`,
  `graph_label_propagation`, `cleannet`, `misdetect_b`, or `ours`.
- `--noise_type`: `symmetric`, `pairflip`, or `instance`.
- `--rho`: label-noise rate, either `0.2` or `0.4`.
- `--seed`: random seed.
- `--gram_variant`: GRAM-only configuration variant. The default is
  `main_adaptive_mixture_uncertainty`.
- `--overwrite`: explicitly allow replacement of files in an existing result
  directory. Existing results are protected by default.

## GRAM Variants

The main GRAM configuration builds separate trace-normalized heat kernels from
the per-epoch margin and last-layer gradient-norm trajectories. After every
query batch it learns simplex-constrained margin, gradient, and identity kernel
weights by Gaussian-surrogate marginal likelihood, then verifies samples with
the largest posterior latent variance. The cold-start weights are
`(1/3, 1/3, 1/3)`. Weight learning starts after at least 30 queried samples and
both clean and mislabeled statuses have been observed; the Gaussian observation
noise variance is fixed at `0.1` so it is not confounded with the identity
kernel weight.
Available variants are:

- `main_adaptive_mixture_uncertainty`: adaptive three-component mixture.
- `abl_fixed_uniform_weights`: fixes all three weights to one third.
- `abl_no_identity`: adaptively learns only margin and gradient weights.
- `abl_margin_only`: uses only the margin trajectory kernel.
- `abl_gradient_only`: uses only the gradient trajectory kernel.
- `abl_k10`, `abl_k25`, `abl_k100`, `abl_k250`: change only the margin and
  gradient trajectory graphs' neighbor count. The diagnostic prior stays at
  `k=50`; kernel ranks, adaptive weights, inference, and variance acquisition
  retain the main settings. Use the existing main result for `k=50`.
- `abl_ucb_beta0`, `abl_ucb_beta0p5`, `abl_ucb_beta1`, `abl_ucb_beta2`: retain
  the main `k=50` adaptive mixture and Gaussian inference, changing only the
  acquisition objective to `mu + beta * sigma` for `beta=0, 0.5, 1, 2`.
  Here `mu` is the Gaussian latent posterior mean and `sigma` is its standard
  deviation (excluding observation noise). `beta=0` selects by mean alone;
  compare these against the main posterior-variance acquisition. Detection
  scores still use the mislabel posterior probability for every variant.

Normal experiment commands omit `--gram_variant`/`--gram-variants` and are
therefore pinned to `main_adaptive_mixture_uncertainty`. The dedicated
`scripts/run_gram_ablations.sh` launcher runs 12 variants by default: four
component/weight ablations, four neighbor counts, and four UCB objectives.
It excludes the main configuration. Neighbor counts and
UCB coefficients are separate sweeps, not a Cartesian product with each other.

For example, run the `k=10` ablation with:

```bash
python scripts/eval_detection.py \
  --dataset cifar10 \
  --method ours \
  --gram_variant abl_k10 \
  --noise_type symmetric \
  --rho 0.2 \
  --seed 0
```

Verification checkpoints are generated automatically from `rho`: `0.001`,
`0.005`, `0.01`, and `0.025`, followed by increments of `0.05` up to `rho`.

### Detection training reuse

Detection runs share one base task-model training record for each exact
`(dataset, noise_type, rho, seed, dataset config, model config)` scenario. The
record contains probabilities, embeddings, loss/AUM statistics, margin and
gradient trajectories, and the trained model state. It is stored under
`results/detection/_shared_training_artifacts/`; a configuration, label, and
training-code fingerprint prevents incompatible reuse. Each baseline and GRAM
still follows its own query trajectory through all configured budgets.

`robust_alc`, `dalc`, and `active_label_cleaning` use that shared record as
their initial state. `active_label_correction` has its own robust-selector
initial state. After each verification checkpoint, these four correction
methods warm-start from the previous model weights with a fresh optimizer.
Active Label Correction also carries forward its contribution weights. Default
warm-start lengths and learning-rate factors are configured in
`config/models.py`: CIFAR models use 2 epochs, tabular models use 10 epochs,
text models use one epoch, and all use 0.1 times the initial learning rate.
The shortened validation configuration trains the initial CIFAR model for 10
epochs. Result `config.json` and
`summary.json` files record the cache fingerprint, cache hit, model-state count,
training invocations, and total effective training epochs.

Two explicitly named compute-control baselines are also available:

- `robust_alc_frozen` keeps the shared backbone fixed while updating its noise
  transition, posterior, and acquisition score from verified labels.
- `dalc_frozen` keeps the shared backbone fixed while retaining DALC's oracle
  and pseudo-label sources and updating its noise posterior.

These are frozen-backbone adaptations, not replacements for the warm-started
`robust_alc` and `dalc` main baselines. Active Label Cleaning and Active Label
Correction do not receive frozen variants because their useful feedback path
depends on updating the underlying model.

## Batch Execution

Quick validation across image, text, and tabular datasets:

```bash
bash scripts/run_detection_experiments.sh \
  --datasets cifar10 atis adult \
  --methods ours \
  --noise-types symmetric \
  --rhos 0.2 \
  --seeds 0 \
  --gpus 0

bash scripts/run_downstream_experiments.sh \
  --datasets cifar10 atis adult \
  --methods ours \
  --noise-types symmetric \
  --rhos 0.2 \
  --seeds 0 \
  --gpus 0
```

Run the default GRAM ablations (the main setting is excluded):

```bash
bash scripts/run_gram_ablations.sh \
  --protocol detection \
  --datasets cifar10 atis adult \
  --noise-types symmetric pairflip instance \
  --rhos 0.2 \
  --seeds 0 1 2 \
  --gpus 0 1
```

Use `--protocol downstream` or `--protocol both` for downstream evaluation,
and `--variants abl_no_identity abl_margin_only` to run only a subset.
To run only the neighbor sweep, pass
`--variants abl_k10 abl_k25 abl_k100 abl_k250`; for only UCB objectives, pass
`--variants abl_ucb_beta0 abl_ucb_beta0p5 abl_ucb_beta1 abl_ucb_beta2`.
Add `main_adaptive_mixture_uncertainty` explicitly to `--variants` if the
matching main result does not exist yet.

Complete detection grid:

```bash
bash scripts/run_detection_experiments.sh \
  --datasets cifar10 cifar100 atis qnli adult letter \
  --methods aum_b el2n_b forgetting_b early_loss_b cleanlab_b knn_label_disagreement_b moderate_b robust_alc robust_alc_frozen dalc dalc_frozen active_label_cleaning active_label_correction graph_label_propagation cleannet misdetect_b ours \
  --noise-types symmetric pairflip instance \
  --rhos 0.2 0.4 \
  --seeds 0 1 2 \
  --gpus 0 1 2 3
```

Use the same options with `run_downstream_experiments.sh` for the complete
downstream grid. Both scripts accept one or more space-separated values for each
option and run their Cartesian product. Jobs are distributed round-robin across
the listed GPUs. The CleanNet baseline uses each modality's shared-backbone
embeddings and supports image, text, and tabular datasets.
