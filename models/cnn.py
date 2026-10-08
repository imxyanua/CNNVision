from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Conv-ReLU-Pool stacks learn local features. Channels stay small for local/Colab training.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # GAP so the classifier does not depend on a fixed spatial size (32, 224, ...).
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Logits only. CrossEntropyLoss applies log-softmax; do not softmax here.
        features = self.features(x)
        pooled = self.pool(features)
        return self.classifier(pooled)

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)

    def settings(self) -> dict[str, int | str]:
        return cnn_settings(num_classes=self.num_classes, in_channels=self.in_channels)


def cnn_settings(num_classes: int, in_channels: int = 3) -> dict[str, int | str]:
    return {
        "architecture": "SimpleCNN",
        "num_classes": num_classes,
        "in_channels": in_channels,
    }


def build_cnn(num_classes: int, in_channels: int = 3) -> SimpleCNN:
    return SimpleCNN(num_classes=num_classes, in_channels=in_channels)


def build_cnn_from_settings(settings: dict[str, int | str]) -> SimpleCNN:
    return SimpleCNN(
        num_classes=int(settings["num_classes"]),
        in_channels=int(settings.get("in_channels", 3)),
    )
