"""Dataset identity and preprocessing settings, not experiment scenarios."""


DATASETS = {
    "cifar10": {
        "modality": "image",
        "source": "torchvision",
        "num_classes": 10,
        "test_metric": "accuracy",
    },
    "cifar100": {
        "modality": "image",
        "source": "torchvision",
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
