from pathlib import Path

import torch
from torchvision import transforms

from preprocessing.image import load_image

DEFAULT_IMAGE_SIZE = 224
DEFAULT_COLOR_MODE = "rgb"
# From-scratch CNN: ToTensor() already scales pixels to [0, 1]. Do not use ImageNet mean/std.
NORMALIZE = "to_tensor_0_1"


def preprocess_settings(
    image_size: int = DEFAULT_IMAGE_SIZE,
    color_mode: str = DEFAULT_COLOR_MODE,
) -> dict[str, int | str | None]:
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    # Store these on the checkpoint later so inference uses the same resize and color mode.
    return {
        "image_size": image_size,
        "color_mode": color_mode,
        "normalize": NORMALIZE,
        "mean": None,
        "std": None,
    }


def build_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    *,
    augment: bool = False,
) -> transforms.Compose:
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    ops: list[object] = []
    if augment:
        # Training only. Eval and inference must not take this branch.
        ops.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2),
            ]
        )
    else:
        # Same output size as training, without random crop/zoom.
        ops.append(transforms.Resize((image_size, image_size)))
    # Converts HWC PIL image to CHW float tensor in [0, 1].
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def build_train_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return build_transforms(image_size, augment=True)


def build_eval_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return build_transforms(image_size, augment=False)


def preprocess_image(
    path: str | Path,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    color_mode: str = DEFAULT_COLOR_MODE,
    augment: bool = False,
) -> torch.Tensor:
    image = load_image(path, color_mode=color_mode)
    tensor = build_transforms(image_size, augment=augment)(image)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Transform pipeline must return a tensor")
    return tensor
