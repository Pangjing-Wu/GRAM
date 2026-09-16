from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from .models import build_model
from .neighbors import pairwise_distance_scale


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelDataset(Dataset):
    def __init__(
        self,
        inputs,
        labels: np.ndarray,
        transform=None,
        source_indices: np.ndarray | None = None,
    ):
        self.inputs = inputs
        labels = np.asarray(labels, dtype=np.int64)
        if source_indices is None:
            self.source_indices = np.arange(len(labels), dtype=np.int64)
            self.labels = labels
        else:
            indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
            if indices.size == 0 or len(np.unique(indices)) != len(indices):
                raise ValueError("training subset must be non-empty and unique")
            if np.any(indices < 0) or np.any(indices >= len(labels)):
                raise IndexError("training subset index is outside the label array")
            self.source_indices = indices
            self.labels = labels[indices]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        source_index = int(self.source_indices[index])
        if isinstance(self.inputs, dict):
            values = {
                key: torch.as_tensor(array[source_index], dtype=torch.long)
                for key, array in self.inputs.items()
            }
        else:
            values = self.inputs[source_index]
            if self.transform is not None:
                values = self.transform(values)
            elif not torch.is_tensor(values):
                values = torch.as_tensor(values, dtype=torch.float32)
        return int(index), values, int(self.labels[index])


@dataclass
class TrainingArtifacts:
    probability: np.ndarray
    embedding: np.ndarray
    early_loss: np.ndarray
    final_loss: np.ndarray
    aum: np.ndarray
    forgetting: np.ndarray
    forgetting_count: np.ndarray
    learned: np.ndarray
    margin_trajectory: np.ndarray
    gradient_trajectory: np.ndarray
    test_accuracy: float
    test_macro_f1: float
    train_seconds: float
    model_state: dict[str, torch.Tensor] | None = None
    training_epochs: int = 0
    warm_started: bool = False
    epoch_offset: int = 0


def warm_start_training_options(config: dict) -> tuple[int, float]:
    epochs = int(config.get("warm_start_epochs", config["epochs"]))
    factor = float(config.get("warm_start_learning_rate_factor", 1.0))
    if epochs <= 0:
        raise ValueError("warm-start epochs must be positive")
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("warm-start learning-rate factor must be positive and finite")
    return epochs, float(config["learning_rate"]) * factor


def _image_transforms(dataset: str):
    from torchvision import transforms

    if dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    else:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    evaluation = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    return train, evaluation


def _move(values, device: torch.device):
    if isinstance(values, dict):
        return {key: tensor.to(device, non_blocking=True) for key, tensor in values.items()}
    return values.to(device, non_blocking=True)


def _logits(model, values):
    return model(**values) if isinstance(values, dict) else model(values)


def _features_and_logits(model, values):
    if isinstance(values, dict):
        features = model.forward_features(**values)
    else:
        features = model.forward_features(values)
    return features, model.classifier(features)


def _training_features_and_logits(model, values):
    """Return the features consumed by the final linear classifier and logits."""
    if isinstance(values, dict):
        features = model.forward_features(**values)
    else:
        features = model.forward_features(values)
    classifier_features = (
        model.dropout(features) if hasattr(model, "dropout") else features
    )
    return classifier_features, model.classifier(classifier_features)


def _batch_plan(
    sample_count: int,
    batch_size: int,
    seed: int,
    epoch: int,
    reference_count: int | None = None,
):
    reference_count = sample_count if reference_count is None else reference_count
    if sample_count <= 0 or reference_count <= 0:
        raise ValueError("sample and reference counts must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    cycles = []
    remaining = reference_count
    while remaining > 0:
        permutation = rng.permutation(sample_count)
        cycles.append(permutation[:remaining])
        remaining -= min(remaining, sample_count)
    order = np.concatenate(cycles)
    return [
        order[start : start + batch_size]
        for start in range(0, reference_count, batch_size)
    ]


def _loader(dataset, *, batch_size=None, batches=None, workers: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    common = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
    }
    if batches is not None:
        return DataLoader(dataset, batch_sampler=batches, **common)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **common)


def _infer(
    model,
    dataset,
    *,
    num_classes: int,
    embedding_dim: int,
    batch_size: int,
    workers: int,
    device: torch.device,
):
    probability = np.empty((len(dataset), num_classes), dtype=np.float32)
    embedding = np.empty((len(dataset), embedding_dim), dtype=np.float32)
    losses = np.empty(len(dataset), dtype=np.float64)
    model.eval()
    loader = _loader(
        dataset, batch_size=batch_size, workers=workers, seed=0
    )
    with torch.no_grad():
        for ids, values, labels in loader:
            positions = ids.numpy()
            values = _move(values, device)
            labels = labels.to(device, non_blocking=True)
            features, logits = _features_and_logits(model, values)
            probability[positions] = torch.softmax(logits, 1).cpu().numpy()
            embedding[positions] = features.float().cpu().numpy()
            losses[positions] = functional.cross_entropy(
                logits, labels, reduction="none"
            ).double().cpu().numpy()
    return probability, embedding, losses


def _infer_losses(
    model,
    dataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> np.ndarray:
    losses = np.empty(len(dataset), dtype=np.float64)
    model.eval()
    loader = _loader(dataset, batch_size=batch_size, workers=workers, seed=0)
    with torch.no_grad():
        for ids, values, labels in loader:
            positions = ids.numpy()
            values = _move(values, device)
            labels = labels.to(device, non_blocking=True)
            losses[positions] = functional.cross_entropy(
                _logits(model, values), labels, reduction="none"
            ).double().cpu().numpy()
    return losses


def _optimizer(model, config: dict, learning_rate: float | None = None):
    name = config["optimizer"]
    active_learning_rate = (
        config["learning_rate"] if learning_rate is None else float(learning_rate)
    )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=active_learning_rate,
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            nesterov=True,
        )
    optimizer_class = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
    return optimizer_class(
        model.parameters(),
        lr=active_learning_rate,
        weight_decay=config["weight_decay"],
    )


def _fit_model(
    data,
    labels: np.ndarray,
    *,
    config: dict,
    seed: int,
    cache_dir: str | Path,
    contribution_config: dict | None,
    train_ids: np.ndarray | None = None,
    training_reference_count: int | None = None,
    noise_transition: np.ndarray | None = None,
    trusted_mask: np.ndarray | None = None,
    initial_model_state: Mapping[str, torch.Tensor] | None = None,
    training_epochs: int | None = None,
    learning_rate: float | None = None,
    epoch_offset: int = 0,
    initial_contribution: np.ndarray | None = None,
):
    started = time.perf_counter()
    robust = contribution_config is not None
    device = resolve_device()
    seed_everything(seed)
    model = build_model(data, config, str(cache_dir))
    if initial_model_state is not None:
        model.load_state_dict(initial_model_state, strict=True)
    model = model.to(device)
    optimizer = _optimizer(model, config, learning_rate)
    active_epochs = (
        config["epochs"] if training_epochs is None else int(training_epochs)
    )
    if active_epochs <= 0:
        raise ValueError("training epochs must be positive")
    epoch_offset = int(epoch_offset)
    if epoch_offset < 0:
        raise ValueError("epoch offset must be non-negative")
    scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config["milestones"], gamma=config["lr_gamma"]
        )
        if config["optimizer"] == "sgd"
        else None
    )
    train_transform = evaluation_transform = None
    if data.modality == "image":
        train_transform, evaluation_transform = _image_transforms(data.name)
    train_set = ModelDataset(
        data.train_x, labels, train_transform, source_indices=train_ids
    )
    evaluation_train_set = ModelDataset(
        data.train_x, labels, evaluation_transform, source_indices=train_ids
    )
    count = len(train_set)
    if training_reference_count is not None and training_reference_count < count:
        raise ValueError("training reference count cannot be smaller than the subset")
    early_sum = np.zeros(count, dtype=np.float64)
    early_count = np.zeros(count, dtype=np.int64)
    aum_sum = np.zeros(count, dtype=np.float64)
    aum_count = np.zeros(count, dtype=np.int64)
    previous_correct = np.zeros(count, dtype=bool)
    learned = np.zeros(count, dtype=bool)
    forgetting_count = np.zeros(count, dtype=np.int64)
    margin_trajectory = np.empty((count, active_epochs), dtype=np.float32)
    gradient_trajectory = np.empty((count, active_epochs), dtype=np.float32)
    contribution = np.full(count, 1.0 / count, dtype=np.float64)
    if initial_contribution is not None:
        if not robust:
            raise ValueError("contribution state is only valid for robust training")
        contribution = np.asarray(initial_contribution, dtype=np.float64).reshape(-1)
        if (
            contribution.shape != (count,)
            or not np.all(np.isfinite(contribution))
            or np.any(contribution < 0.0)
            or contribution.sum() <= 0.0
        ):
            raise ValueError(
                "initial contribution state must be finite and non-negative"
            )
        contribution = contribution / contribution.sum()
    transition_tensor = None
    local_trusted = None
    if noise_transition is not None:
        transition = np.asarray(noise_transition, dtype=np.float64)
        expected = (data.num_classes, data.num_classes)
        if transition.shape != expected or np.any(transition < 0):
            raise ValueError(f"noise transition must have shape {expected} and be non-negative")
        row_sum = transition.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise ValueError("every noise-transition row must have positive mass")
        transition = transition / row_sum
        transition_tensor = torch.as_tensor(
            transition, dtype=torch.float32, device=device
        )
        full_trusted = (
            np.zeros(len(labels), dtype=bool)
            if trusted_mask is None
            else np.asarray(trusted_mask, dtype=bool).reshape(-1)
        )
        if full_trusted.shape != (len(labels),):
            raise ValueError("trusted mask and labels must be aligned")
        local_trusted = (
            full_trusted
            if train_ids is None
            else full_trusted[np.asarray(train_ids, dtype=np.int64)]
        )

    for local_epoch in range(1, active_epochs + 1):
        epoch = epoch_offset + local_epoch
        epoch_margin_sum = np.zeros(count, dtype=np.float64)
        epoch_gradient_sum = np.zeros(count, dtype=np.float64)
        epoch_observation_count = np.zeros(count, dtype=np.int64)
        epoch_seed = int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0])
        seed_everything(epoch_seed)
        batches = _batch_plan(
            count,
            config["batch_size"],
            seed,
            epoch,
            reference_count=training_reference_count,
        )
        loader = _loader(
            train_set,
            batches=batches,
            workers=config["num_workers"],
            seed=epoch_seed,
        )
        model.train()
        for ids, values, batch_labels in loader:
            positions = ids.numpy()
            values = _move(values, device)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            classifier_features, logits = _training_features_and_logits(model, values)
            ordinary_losses = functional.cross_entropy(
                logits, batch_labels, reduction="none"
            )
            if transition_tensor is None:
                losses = ordinary_losses
            else:
                noisy_probability = torch.softmax(logits, dim=1) @ transition_tensor
                losses = -torch.log(
                    noisy_probability[
                        torch.arange(len(batch_labels), device=device), batch_labels
                    ].clamp_min(1.0e-12)
                )
                trusted = torch.as_tensor(
                    local_trusted[positions], dtype=torch.bool, device=device
                )
                losses = torch.where(trusted, ordinary_losses, losses)
            if local_epoch <= config["early_epochs"]:
                np.add.at(early_sum, positions, ordinary_losses.detach().double().cpu().numpy())
                np.add.at(early_count, positions, 1)

            detached_logits = logits.detach().float().cpu().numpy()
            detached_labels = batch_labels.detach().cpu().numpy().astype(np.int64)
            row = np.arange(len(detached_labels))
            assigned = detached_logits[row, detached_labels]
            alternative = detached_logits.copy()
            alternative[row, detached_labels] = -np.inf
            margin = assigned - alternative.max(axis=1)
            np.add.at(aum_sum, positions, margin)
            np.add.at(aum_count, positions, 1)
            detached_probability = torch.softmax(
                logits.detach(), dim=1
            ).float().cpu().numpy()
            residual = detached_probability
            residual[row, detached_labels] -= 1.0
            detached_features = (
                classifier_features.detach().float().cpu().numpy()
            )
            gradient_norm = np.linalg.norm(residual, axis=1) * np.sqrt(
                1.0 + np.einsum("ij,ij->i", detached_features, detached_features)
            )
            np.add.at(epoch_margin_sum, positions, margin)
            np.add.at(epoch_gradient_sum, positions, gradient_norm)
            np.add.at(epoch_observation_count, positions, 1)
            correct = detached_logits.argmax(axis=1) == detached_labels
            forgetting_count[positions] += previous_correct[positions] & ~correct
            learned[positions] |= correct
            previous_correct[positions] = correct
            if robust:
                weights = torch.as_tensor(
                    contribution[positions], dtype=losses.dtype, device=device
                )
                weights /= weights.mean()
                loss = (weights * losses).mean()
            else:
                loss = losses.mean()
            loss.backward()
            optimizer.step()

        if np.any(epoch_observation_count == 0):
            raise RuntimeError("trajectory collection missed a training sample")
        margin_trajectory[:, local_epoch - 1] = (
            epoch_margin_sum / epoch_observation_count
        )
        gradient_trajectory[:, local_epoch - 1] = (
            epoch_gradient_sum / epoch_observation_count
        )

        if robust:
            epoch_losses = _infer_losses(
                model,
                evaluation_train_set,
                batch_size=config["batch_size"],
                workers=config["num_workers"],
                device=device,
            )
            scale_seed = int(
                np.random.SeedSequence([seed, epoch, 4201]).generate_state(1)[0]
            )
            sigma = pairwise_distance_scale(
                epoch_losses,
                seed=scale_seed,
                pair_count=contribution_config["distance_pairs"],
            )
            log_weight = -np.square(epoch_losses) / sigma
            # Keep at least one exponential at one to avoid all-zero underflow.
            update = np.exp(log_weight - log_weight.max())
            update /= update.sum()
            delta = contribution_config["contribution_step_size"]
            contribution = (1.0 - delta) * contribution + delta * update
            contribution /= contribution.sum()
        if scheduler is not None:
            scheduler.step()

    return (
        model,
        evaluation_train_set,
        early_sum / np.maximum(early_count, 1),
        aum_sum / np.maximum(aum_count, 1),
        forgetting_count,
        learned,
        margin_trajectory,
        gradient_trajectory,
        contribution,
        device,
        time.perf_counter() - started,
        evaluation_transform,
    )


def train_task_model(
    data,
    labels: np.ndarray,
    *,
    config: dict,
    seed: int,
    cache_dir: str | Path,
    train_ids: np.ndarray | None = None,
    training_reference_count: int | None = None,
    evaluate_test: bool = True,
    noise_transition: np.ndarray | None = None,
    trusted_mask: np.ndarray | None = None,
    initial_model_state: Mapping[str, torch.Tensor] | None = None,
    training_epochs: int | None = None,
    learning_rate: float | None = None,
    capture_model_state: bool = False,
    epoch_offset: int = 0,
) -> TrainingArtifacts:
    (
        model,
        evaluation_train_set,
        early_loss,
        aum,
        forgetting_count,
        learned,
        margin_trajectory,
        gradient_trajectory,
        _,
        device,
        train_seconds,
        evaluation_transform,
    ) = _fit_model(
        data,
        labels,
        config=config,
        seed=seed,
        cache_dir=cache_dir,
        contribution_config=None,
        train_ids=train_ids,
        training_reference_count=training_reference_count,
        noise_transition=noise_transition,
        trusted_mask=trusted_mask,
        initial_model_state=initial_model_state,
        training_epochs=training_epochs,
        learning_rate=learning_rate,
        epoch_offset=epoch_offset,
    )
    probability, embedding, final_loss = _infer(
        model,
        evaluation_train_set,
        num_classes=data.num_classes,
        embedding_dim=model.embedding_dim,
        batch_size=config["batch_size"],
        workers=config["num_workers"],
        device=device,
    )
    if evaluate_test:
        test_set = ModelDataset(data.test_x, data.test_y, evaluation_transform)
        test_probability, _, _ = _infer(
            model,
            test_set,
            num_classes=data.num_classes,
            embedding_dim=model.embedding_dim,
            batch_size=config["batch_size"],
            workers=config["num_workers"],
            device=device,
        )
        prediction = test_probability.argmax(axis=1)
        accuracy = float((prediction == data.test_y).mean())
        macro_f1 = float(
            f1_score(data.test_y, prediction, average="macro", zero_division=0)
        )
    else:
        accuracy = macro_f1 = float("nan")
    return TrainingArtifacts(
        probability=probability,
        embedding=embedding,
        early_loss=early_loss,
        final_loss=final_loss,
        aum=aum,
        forgetting=np.where(
            learned,
            forgetting_count,
            forgetting_count.max(initial=0) + 1,
        ).astype(np.float64),
        forgetting_count=forgetting_count,
        learned=learned,
        margin_trajectory=margin_trajectory,
        gradient_trajectory=gradient_trajectory,
        test_accuracy=accuracy,
        test_macro_f1=macro_f1,
        train_seconds=train_seconds,
        model_state=(
            {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            if capture_model_state
            else None
        ),
        training_epochs=(
            int(config["epochs"])
            if training_epochs is None
            else int(training_epochs)
        ),
        warm_started=initial_model_state is not None,
        epoch_offset=int(epoch_offset),
    )


def train_downstream_model(
    data,
    labels: np.ndarray,
    train_ids: np.ndarray,
    *,
    config: dict,
    seed: int,
    cache_dir: str | Path,
) -> TrainingArtifacts:
    """Train a fresh downstream model with method-independent update count."""
    return train_task_model(
        data,
        labels,
        config=config,
        seed=seed,
        cache_dir=cache_dir,
        train_ids=train_ids,
        training_reference_count=data.sample_count,
    )
