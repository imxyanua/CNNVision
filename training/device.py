from __future__ import annotations

import torch


def select_device(name: str | None = None) -> torch.device:
    if name:
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
