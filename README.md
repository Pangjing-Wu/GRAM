# Label Verification Benchmark

This project provides two experiment protocols:

- `detection` evaluates label-issue rankings on the full training pool and the
  currently unverified subset using AUPRC and AUROC.
- `downstream` repairs or filters training data based on method predictions and
  evaluates a downstream model on the clean test set.

Downstream evaluation reuses the matching detection query trajectory and
per-checkpoint predictions. It never selects samples, calls an expert oracle,
or changes detection outputs. Downstream models are trained from scratch.

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
`summary.json`, `config.json`, and `downstream_snapshots.npz`. The latter stores
scores, binary issue decisions, verified-sample masks, and repaired labels for
every budget checkpoint so downstream evaluation can reuse the exact trajectory.
GRAM runs additionally write
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

Each run saves only **`downstream_metrics.csv` and `config.json`**. Progress goes
to the terminal. Numerical rows are saved atomically after each model finishes;
`config.json` has `status=complete` only after all requested rows are saved.

Budgets are read from the existing detection configuration. At **every saved
budget checkpoint**, the experiment trains two fresh models:

- `repair_only`: apply the cumulative expert labels from `queries.csv` and keep
  all training samples.
- `repair_otsu`: apply the same expert labels, compute an exact Otsu threshold
  from **all samples' scores at that checkpoint**, then delete only unverified
  samples whose scores are strictly above that threshold. Verified samples are
  always retained. There is no deletion-rate or histogram-bin hyperparameter;
  constant scores retain the whole pool.

`noisy` and `clean` references are separate tasks and need no detection results:

```bash
python scripts/eval_downstream.py --dataset cifar10 --reference noisy \
  --noise_type symmetric --rho 0.2 --seed 0
python scripts/eval_downstream.py --dataset cifar10 --reference clean --seed 0
```

The noisy reference uses the full training set with the original injected labels,
without repair or filtering. It runs once per dataset/noise type/rho/seed, shared
by all methods and GRAM variants. The clean reference uses the original clean
labels and runs once per dataset/seed, shared across noise types and ratios.
Their outputs are stored at:

```text
results/downstream/<dataset>/<noise_type>/rho<rho>/noisy/seed<seed>/
results/downstream/<dataset>/clean/seed<seed>/
```

Each reference CSV contains one checkpoint-0 row; the clean reference has an
empty budget field. Method tasks only train `repair_only` and `repair_otsu`.
All models share
the downstream initialization seed and the same update count, based on the
original training-set size. DALC's internal pseudo labels are replayed for
scoring only; they do not become expert-repaired downstream labels.

The CSV contains test accuracy/macro-F1, task performance, expert repair counts,
Otsu thresholds, deletion counts/precision/recall, retained fractions, remaining
label noise, and training times. `config.json` records the source detection
directory, effective settings, and how checkpoint scores were obtained.

When `downstream_snapshots.npz` exists, scores are read directly. For legacy
results without snapshots, scores are recovered by replaying the **saved expert
and pseudo-label logs** against the historical training artifact. GRAM and
frozen-backbone baselines need inference only. Warm-start baselines replay their
classifier updates; Active Label Correction also reconstructs its robust
selector. No acquisition selection or expert verification is repeated. Recovered
final scores must match `predictions.csv` before downstream training starts.
Missing source results, incompatible settings, corrupt snapshots, or unavailable
historical artifacts produce an error; detection is never automatically rerun.

By default all saved checkpoints are evaluated. To evaluate just existing 1%
and 5% checkpoints, add `--checkpoints 0.01 0.05`; the original query schedule
is still used when recovering scores. `--data-root` and `--results-root` override
the configured roots. `--overwrite` replaces downstream outputs only.

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
  `main_adaptive_mixture`.
- `--overwrite`: explicitly allow replacement of files in an existing result
  directory. Existing results are protected by default.

## GRAM Variants

The main GRAM configuration builds separate trace-normalized heat kernels from
margin and last-layer gradient-norm trajectories. After each query batch, it
learns simplex-constrained margin, gradient, and identity kernel weights by
Gaussian-surrogate marginal likelihood. Verification maximizes

```text
a_t(i) = p_t(i) + beta * IV_t(i)
p_t(i) = Phi(mu_t(i) / sqrt(Sigma_t[i,i] + tau^2))
IV_t(i) = ||Sigma_t[U_t,i]||_2^2 / (|U_t| * (Sigma_t[i,i] + tau^2))
```

Here `p_t(i)` is the existing mislabel posterior score and represents expected
direct repair value. `IV_t(i)` is the mean latent posterior variance reduction
over **all currently unverified samples, including i**. It includes both the
self reduction and covariance-mediated propagation. Candidate masks restrict
selection, not the information pool. The calculation conditions on one more
Gaussian observation with the current kernel weights held fixed; it does not
predict the effect of relearning weights. Each batch ranks a common posterior.

The default `beta=1` is a starting setting, not a validated optimum. IV is used
in its raw mean-variance units, without rank/min-max normalization or a separate
uncertainty term. Low-rank factors compute IV in O(|U_t| r^2) time and
O(|U_t| r + r^2) working memory, without an N-by-N covariance matrix. Cold-start
weights remain `(1/3, 1/3, 1/3)`. Weight learning starts after at least 30 verified
samples and both statuses have appeared; observation variance remains `0.1`.
The final mislabel scoring rule and binary feedback to the GP are unchanged.
In detection, direct repair value measures mislabels confirmed for correction;
the downstream protocol actually applies the oracle's corrected labels.
GRAM query logs include `repair_value` and `informative_value`; the local-only
control records the local IV used by its acquisition, and controls without an
IV term leave that field empty.

Current variants:

- `main_adaptive_mixture`: adaptive three-component mixture, `p + IV`.
- `abl_fixed_uniform_weights`: fix all three weights to one third.
- `abl_no_identity`: adaptive margin + gradient.
- `abl_no_gradient`: adaptive margin + identity.
- `abl_no_margin`: adaptive gradient + identity.
- `abl_k10`, `abl_k25`, `abl_k100`, `abl_k250`: change trajectory
  graph neighbors only; diagnostic neighbors remain 50. Main supplies `k=50`.
- `abl_risk_only`: posterior risk `p` (equivalent to `beta=0`, not latent mean).
- `abl_iv_only`: IV alone.
- `abl_variance_only`: the old posterior-variance acquisition.
- `abl_local_only`: `p + beta * Sigma[i,i]^2 / (|U| * (Sigma[i,i] + tau^2))`;
  remove cross-sample propagation while retaining the same self-reduction term.
- `abl_beta0p1`, `abl_beta10`, `abl_beta100`, `abl_beta1000`, `abl_beta10000`:
  vary only IV's coefficient.
  Main supplies `beta=1` and risk-only supplies `beta=0`.

`scripts/run_detection_experiments.sh` uses the shared detection parameters
from the former grid launcher. By default it crosses beta with k for the full
adaptive kernel and repair+IV; beta=0 uses the equivalent risk-only configuration.

| Parameter | Supported values |
| --- | --- |
| `--datasets` | cifar10, cifar100, atis, qnli, adult, letter |
| `--noise-types` | symmetric, pairflip, instance |
| `--rhos` | 0.2, 0.4 |
| `--neighbors` | 10, 25, 50, 100, 250 (all by default) |
| `--betas` | 0, 0.1, 1, 10, 100, 1000, 10000 (all by default) |

This gives **35 GRAM configurations per dataset/noise/rho/seed scenario**.
Baselines run once per scenario and seed. Kernel and acquisition ablations are
available through explicit `--gram-variants`; that option cannot be combined
with `--betas` or `--neighbors`. The separate grid Python and Shell launchers
have been removed.

Observation variance and weight regularization remain 0.1; diagnostic neighbors
remain 50. Equivalent configurations keep their historical `main_adaptive_mixture`,
`abl_*`, or `grid_*` names so completed results can be reused in place. Every
effective parameter is saved in `config.json`. Completed runs are skipped by
the evaluator; incomplete output directories require `--overwrite`.
See [the ablation rationale](docs/gram_repair_iv_ablations.md),
[the analysis notebook](analysis/ablation.ipynb), and
[historical grid timing evidence](docs/gram_grid_runtime.md).

For example, run the new `k=10` ablation with:

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
  --neighbors 50 --betas 1 \
  --noise-types symmetric \
  --rhos 0.2 \
  --seeds 0 \
  --gpus 0

bash scripts/run_downstream_experiments.sh \
  --datasets cifar10 atis adult \
  --methods ours \
  --gram-variants main_adaptive_mixture \
  --noise-types symmetric \
  --rhos 0.2 \
  --seeds 0 \
  --gpus 0
```

Preview the detection commands without starting experiments:

```bash
bash scripts/run_detection_experiments.sh \
  --datasets cifar10 cifar100 atis qnli adult letter \
  --methods ours \
  --noise-types symmetric pairflip instance \
  --rhos 0.2 0.4 \
  --neighbors 10 25 50 100 250 \
  --betas 0 0.1 1 10 100 1000 10000 \
  --seeds 0 1 2 \
  --gpus 0 1 \
  --dry-run
```

Remove `--dry-run` to launch. Omit `--neighbors` and `--betas` to use all of
their supported values, or restrict them, for example with
`--neighbors 10 50 --betas 1 10000`. The full GRAM batch contains
35 × 6 × 3 × 2 × 3 = **3,780 runs**, before skipping completed results.

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

The downstream batch discovers existing completed detection scenarios by default:

```bash
bash scripts/run_downstream_experiments.sh --gpus 0 1 --dry-run
```

Remove `--dry-run` to launch. Dataset, method, noise, rho, and seed options are
optional filters over existing results. For GRAM, it discovers the available
betas and automatically chooses k for each `dataset / noise_type / rho / beta`
using the shared best-k interface, then evaluates the selected variant's
available complete seeds. Baselines use each existing complete detection run
once. The batch also deduplicates noisy and clean references across these
selected scenarios. References and baselines finish before GRAM tasks start.
Completed downstream results are skipped unless `--overwrite` is supplied.

Downstream defaults to **4 independent processes per GPU**. Workers take the
next job from a shared queue as soon as they finish, so task durations do not
leave a GPU waiting for another worker's fixed assignment. Set
`--workers-per-gpu N` to change concurrency; `1` restores one process per GPU.
For the seed0 baseline and beta=10/100 batch on two GPUs:

```bash
bash scripts/run_downstream_experiments.sh \
  --seeds 0 --betas 10 100 --gpus 0 1 --workers-per-gpu 4
```

This permits 8 simultaneous downstream processes. Each process keeps the same
model, training configuration, and automatic k selection. A failed task is
reported and other queued tasks continue; the batch exits nonzero if any task
fails. Interrupting the batch terminates its active task process groups.

Use `--betas 1 100` to filter beta, or `--gram-variants ...` to explicitly
choose saved configurations, including ablations; these options are mutually
exclusive. The automatic selector requires completed seed0 for all five k
candidates; partially finished beta/k groups are reported and skipped.
If `--seeds` is supplied, the selected variant must contain every requested
seed or planning fails before starting jobs. Missing seeds never change k.

The shared, read-only selection interface is:

```bash
python3 scripts/best_k.py --dataset qnli --noise-type instance --rho 0.4 --beta 10000
# Add --format k or --format variant for one scalar; --format tsv supports batches.
```

Python callers use
`scripts.best_k.get_best_k(dataset, noise_type, rho, beta, results_root=...)`.
The same module supplies budget integration, seed/noise means, and selection to
`analysis/detect_result_vis.ipynb`. It uses only full adaptive GRAM beta × k
detection results, averages each configuration's available complete seeds,
and maximizes budget-average AUPRC + Precision. On exact ties it compares
AUPRC, then Precision, then prefers smaller k. It never chooses beta or
uses downstream/test scores. The plotting API also supports `noise_mean`
(best k after averaging noises) and `global` (one k per beta across datasets
and noise rates); the isolated numeric notebook remains a fixed snapshot.
The historical beta=1 ablation reference in `docs/gram_k_selection.md`
has a different common-window AUPRC-only rule and is not the downstream selector.

Both scripts accept one or more space-separated values for list options.
Detection expands the requested parameter combinations; downstream filters
existing results and selects k automatically. Detection distributes jobs round-robin;
downstream uses the shared queue with the configured number of workers per GPU.
The CleanNet baseline uses each modality's shared-backbone
embeddings and supports image, text, and tabular datasets.
