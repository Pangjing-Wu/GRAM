"""Dataset identity and preprocessing settings, not experiment scenarios."""


DATASETS = {
    "cifar10": {
        "modality": "image",
        "source": "uoft-cs/cifar10",
        "data_files": {
            "train": "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/refs%2Fconvert%2Fparquet/plain_text/train/0000.parquet",
            "test": "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/refs%2Fconvert%2Fparquet/plain_text/test/0000.parquet",
        },
        "image_column": "img",
        "label_column": "label",
        "num_classes": 10,
        "test_metric": "accuracy",
    },
    "cifar100": {
        "modality": "image",
        "source": "uoft-cs/cifar100",
        "data_files": {
            "train": "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/refs%2Fconvert%2Fparquet/cifar100/train/0000.parquet",
            "test": "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/refs%2Fconvert%2Fparquet/cifar100/test/0000.parquet",
        },
        "image_column": "img",
        "label_column": "fine_label",
        "num_classes": 100,
        "test_metric": "accuracy",
    },
    "atis": {
        "modality": "text",
        "source": "pfsv/atis",
        "num_classes": 17,
        "test_metric": "macro_f1",
        "text_columns": ("text",),
    },
    "qnli": {
        "modality": "text",
        "source": "nyu-mll/glue",
        "subset": "qnli",
        "num_classes": 2,
        "test_metric": "accuracy",
        "text_columns": ("question", "sentence"),
    },
    "adult": {
        "modality": "tabular",
        "source": "openml:adult:2",
        "num_classes": 2,
        "test_metric": "macro_f1",
    },
    "letter": {
        "modality": "tabular",
        "source": "openml:letter:1",
        "num_classes": 26,
        "test_metric": "macro_f1",
    },
}
