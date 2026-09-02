from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == channels
            else nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        )

    def forward(self, values):
        residual = self.shortcut(values)
        values = torch.relu(self.bn1(self.conv1(values)))
        values = self.bn2(self.conv2(values))
        return torch.relu(values + residual)


class CifarResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        channels = (64, 128, 256, 512)
        blocks, previous = [], 64
        for stage, channel in enumerate(channels):
            stride = 1 if stage == 0 else 2
            blocks.extend(
                [ResidualBlock(previous, channel, stride), ResidualBlock(channel, channel)]
            )
            previous = channel
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        self.embedding_dim = 512
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, ResidualBlock):
                nn.init.zeros_(module.bn2.weight)

    def forward_features(self, values):
        values = self.blocks(self.stem(values))
        return self.pool(values).flatten(1)

    def forward(self, values):
        return self.classifier(self.forward_features(values))


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], num_classes: int):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for dimension in hidden_dims:
            layers.extend([nn.Linear(previous, dimension), nn.ReLU(inplace=True)])
            previous = dimension
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(previous, num_classes)
        self.embedding_dim = previous

    def forward_features(self, values):
        return self.features(values)

    def forward(self, values):
        return self.classifier(self.forward_features(values))


class RobertaClassifier(nn.Module):
    def __init__(self, name: str, num_classes: int, cache_dir: str):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(
            name,
            cache_dir=cache_dir,
            add_pooling_layer=False,
        )
        self.embedding_dim = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(float(self.encoder.config.hidden_dropout_prob))
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward_features(self, **inputs):
        return self.encoder(**inputs).last_hidden_state[:, 0]

    def forward(self, **inputs):
        return self.classifier(self.dropout(self.forward_features(**inputs)))


def build_model(data, config: dict, cache_dir: str) -> nn.Module:
    if data.modality == "image":
        return CifarResNet18(data.num_classes)
    if data.modality == "text":
        return RobertaClassifier(config["model"], data.num_classes, cache_dir)
    return TabularMLP(
        data.metadata["input_dim"], config["hidden_dims"], data.num_classes
    )
