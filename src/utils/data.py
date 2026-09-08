from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .local_data import ensure_model_cached, load_dataset_or_download, prepare_dataset


class ImageInputs:
    def __init__(self, source, image_column: str | None = None):
        self.source = source
        self.image_column = image_column

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        row = self.source[int(index)]
        return row[self.image_column] if self.image_column is not None else row[0]


@dataclass
class TaskData:
    name: str
    modality: str
    train_x: Any
    test_x: Any
    clean_train_y: np.ndarray
    test_y: np.ndarray
    noise_x: Any
    num_classes: int
    test_metric: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.clean_train_y = np.asarray(self.clean_train_y, dtype=np.int64)
        self.test_y = np.asarray(self.test_y, dtype=np.int64)
        if self._input_count(self.train_x) != len(self.clean_train_y):
            raise ValueError("training inputs and labels have different lengths")
        if self._input_count(self.test_x) != len(self.test_y):
            raise ValueError("test inputs and labels have different lengths")
        if self.clean_train_y.min() < 0 or self.clean_train_y.max() >= self.num_classes:
            raise ValueError("training labels are outside the configured class range")
        if self.test_y.min() < 0 or self.test_y.max() >= self.num_classes:
            raise ValueError("test labels are outside the configured class range")

    @property
    def sample_count(self) -> int:
        return len(self.clean_train_y)

    @staticmethod
    def _input_count(inputs) -> int:
        if isinstance(inputs, dict):
            counts = {len(values) for values in inputs.values()}
            if len(counts) != 1:
                raise ValueError("encoded text fields have inconsistent lengths")
            return counts.pop()
        return len(inputs)


def _pooled_cifar_features(images: np.ndarray) -> np.ndarray:
    values = np.asarray(images, dtype=np.float32) / 255.0
    return values.reshape(-1, 8, 4, 8, 4, 3).mean((2, 4)).reshape(len(values), -1)


def _pooled_huggingface_image_features(dataset, image_column: str) -> np.ndarray:
    features = np.empty((len(dataset), 8 * 8 * 3), dtype=np.float32)
    for start in range(0, len(dataset), 1000):
        stop = min(start + 1000, len(dataset))
        images = np.stack(
            [np.asarray(dataset[index][image_column]) for index in range(start, stop)]
        )
        features[start:stop] = _pooled_cifar_features(images)
    return features


def _load_cifar(name: str, root: Path, config: dict) -> TaskData:
    raw = load_dataset_or_download(name, root, config)
    train, test = raw["train"], raw["test"]
    image_column = config["image_column"]
    label_column = config["label_column"]
    return TaskData(
        name=name,
        modality="image",
        train_x=ImageInputs(train, image_column),
        test_x=ImageInputs(test, image_column),
        clean_train_y=np.asarray(train[label_column]),
        test_y=np.asarray(test[label_column]),
        noise_x=_pooled_huggingface_image_features(train, image_column),
        num_classes=config["num_classes"],
        test_metric=config["test_metric"],
    )


def _encode_texts(
    first: list[str],
    second: list[str] | None,
    *,
    dataset_name: str,
    model_name: str,
    max_length: int,
    cache_dir: Path,
) -> dict[str, np.ndarray]:
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=str(cache_dir), local_files_only=True,
        )
    except OSError:
        prepare_dataset(dataset_name, cache_dir.parent)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=str(cache_dir), local_files_only=True,
        )
    encoded = tokenizer(
        first,
        text_pair=second,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    return {key: np.asarray(value, dtype=np.int64) for key, value in encoded.items()}


def _hashed_text_features(first: list[str], second: list[str] | None):
    from sklearn.feature_extraction.text import HashingVectorizer

    texts = first if second is None else [f"{a} [SEP] {b}" for a, b in zip(first, second)]
    return HashingVectorizer(
        n_features=256,
        alternate_sign=True,
        norm="l2",
    ).transform(texts)


def _load_atis(root: Path, config: dict, model_config: dict) -> TaskData:
    raw = load_dataset_or_download("atis", root, config)
    train_rows = [row for row in raw["train"] if "+" not in row["intent"]]
    class_names = sorted({row["intent"] for row in train_rows})
    if len(class_names) != config["num_classes"]:
        raise ValueError(f"ATIS expected 17 train intents, found {len(class_names)}")
    label_id = {name: index for index, name in enumerate(class_names)}
    test_rows = [
        row
        for row in raw["test"]
        if "+" not in row["intent"] and row["intent"] in label_id
    ]
    train_text = [row["text"] for row in train_rows]
    test_text = [row["text"] for row in test_rows]
    cache = root / "huggingface"
    return TaskData(
        name="atis",
        modality="text",
        train_x=_encode_texts(
            train_text,
            None,
            dataset_name="atis",
            model_name=model_config["model"],
            max_length=model_config["max_length"],
            cache_dir=cache,
        ),
        test_x=_encode_texts(
            test_text,
            None,
            dataset_name="atis",
            model_name=model_config["model"],
            max_length=model_config["max_length"],
            cache_dir=cache,
        ),
        clean_train_y=np.asarray([label_id[row["intent"]] for row in train_rows]),
        test_y=np.asarray([label_id[row["intent"]] for row in test_rows]),
        noise_x=_hashed_text_features(train_text, None),
        num_classes=config["num_classes"],
        test_metric=config["test_metric"],
        metadata={"class_names": class_names},
    )


def _load_qnli(root: Path, config: dict, model_config: dict) -> TaskData:
    raw = load_dataset_or_download("qnli", root, config)
    train, test = raw["train"], raw["validation"]
    train_first, train_second = train["question"], train["sentence"]
    test_first, test_second = test["question"], test["sentence"]
    cache = root / "huggingface"
    return TaskData(
        name="qnli",
        modality="text",
        train_x=_encode_texts(
            train_first,
            train_second,
            dataset_name="qnli",
            model_name=model_config["model"],
            max_length=model_config["max_length"],
            cache_dir=cache,
        ),
        test_x=_encode_texts(
            test_first,
            test_second,
            dataset_name="qnli",
            model_name=model_config["model"],
            max_length=model_config["max_length"],
            cache_dir=cache,
        ),
        clean_train_y=np.asarray(train["label"]),
        test_y=np.asarray(test["label"]),
        noise_x=_hashed_text_features(train_first, train_second),
        num_classes=config["num_classes"],
        test_metric=config["test_metric"],
    )


def _split_and_preprocess_adult(root: Path, seed: int, config: dict):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    bunch = load_dataset_or_download("adult", root, config)
    frame = bunch.data
    target = bunch.target.astype(str).str.strip().str.rstrip(".")
    labels = (target == ">50K").astype(np.int64).to_numpy()
    train_x, test_x, train_y, test_y = train_test_split(
        frame, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    categorical = frame.select_dtypes(include=["object", "category", "bool"]).columns
    numerical = [column for column in frame.columns if column not in categorical]
    processor = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numerical,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                list(categorical),
            ),
        ],
        sparse_threshold=0.0,
    )
    return (
        processor.fit_transform(train_x).astype(np.float32),
        processor.transform(test_x).astype(np.float32),
        train_y,
        test_y,
    )


def _load_tabular(name: str, root: Path, config: dict, seed: int) -> TaskData:
    if name == "adult":
        train_x, test_x, train_y, test_y = _split_and_preprocess_adult(root, seed, config)
        metadata = {}
    else:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        bunch = load_dataset_or_download(name, root, config)
        values = np.asarray(bunch.data, dtype=np.float32)
        encoder = LabelEncoder()
        labels = encoder.fit_transform(bunch.target).astype(np.int64)
        train_x, test_x, train_y, test_y = train_test_split(
            values, labels, test_size=0.2, random_state=seed, stratify=labels
        )
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x).astype(np.float32)
        test_x = scaler.transform(test_x).astype(np.float32)
        metadata = {"class_names": encoder.classes_.tolist()}
    return TaskData(
        name=name,
        modality="tabular",
        train_x=train_x,
        test_x=test_x,
        clean_train_y=train_y,
        test_y=test_y,
        noise_x=train_x,
        num_classes=config["num_classes"],
        test_metric=config["test_metric"],
        metadata={"input_dim": int(train_x.shape[1]), **metadata},
    )


def load_data(
    name: str,
    *,
    root: str | Path,
    seed: int,
    dataset_config: dict,
    model_config: dict,
) -> TaskData:
    root = Path(root)
    modality = dataset_config["modality"]
    if modality == "image":
        return _load_cifar(name, root, dataset_config)
    if modality == "text":
        ensure_model_cached(name, root, model_config["model"])
        if name == "atis":
            return _load_atis(root, dataset_config, model_config)
        return _load_qnli(root, dataset_config, model_config)
    return _load_tabular(name, root, dataset_config, seed)
