"""Shared train and eval transforms. Augmentation is training-only."""

from pathlib import Path

import torch
from torchvision import transforms

from preprocessing.image import load_image

DEFAULT_IMAGE_SIZE = 224
DEFAULT_COLOR_MODE = "rgb"
NORMALIZE = "to_tensor_0_1"


def preprocess_settings(
    image_size: int = DEFAULT_IMAGE_SIZE,
    color_mode: str = DEFAULT_COLOR_MODE,
) -> dict[str, int | str | None]:
    """Settings that must be reused at validation, test, and inference."""
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
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
    """Build the pipeline that turns a PIL image into a float tensor in [0, 1].

    Train (augment=True): random crop/zoom, horizontal flip, small rotation, brightness.
    Eval and inference (augment=False): resize only, then ToTensor.
    """
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    ops: list[object] = []
    if augment:
        ops.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2),
            ]
        )
    else:
        ops.append(transforms.Resize((image_size, image_size)))
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
    """Load one image and apply the shared transforms. Inference must use augment=False."""
    image = load_image(path, color_mode=color_mode)
    tensor = build_transforms(image_size, augment=augment)(image)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Transform pipeline must return a tensor")
    return tensor
