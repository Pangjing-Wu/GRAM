"""Task-model settings shared by all query policies."""


_IMAGE = {
    "model": "resnet18",
    "epochs": 10,
    "early_epochs": 2,
    "batch_size": 128,
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "weight_decay": 5.0e-4,
    "momentum": 0.9,
    "milestones": (5, 8),
    "lr_gamma": 0.1,
    "num_workers": 4,
    "warm_start_epochs": 2,
    "warm_start_learning_rate_factor": 0.1,
}

_TEXT = {
    "model": "FacebookAI/roberta-base",
    "epochs": 3,
    "early_epochs": 1,
    "batch_size": 32,
    "optimizer": "adamw",
    "learning_rate": 2.0e-5,
    "weight_decay": 0.01,
    "max_length": 128,
    "num_workers": 2,
    "warm_start_epochs": 1,
    "warm_start_learning_rate_factor": 0.1,
}

_TABULAR = {
    "model": "mlp",
    "epochs": 50,
    "early_epochs": 10,
    "batch_size": 256,
    "optimizer": "adam",
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-4,
    "hidden_dims": (256, 128, 64),
    "num_workers": 0,
    "warm_start_epochs": 10,
    "warm_start_learning_rate_factor": 0.1,
}

MODELS = {
    "cifar10": dict(_IMAGE),
    "cifar100": dict(_IMAGE),
    "atis": dict(_TEXT),
    "qnli": dict(_TEXT),
    "adult": dict(_TABULAR),
    "letter": dict(_TABULAR),
}
