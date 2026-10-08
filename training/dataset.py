from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from preprocessing.image import load_image
from preprocessing.transforms import (
    DEFAULT_COLOR_MODE,
    DEFAULT_IMAGE_SIZE,
    build_eval_transforms,
    build_train_transforms,
)
from training.split import DEFAULT_SEED, DatasetSplit, prepare_dataset


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform,
        color_mode: str = DEFAULT_COLOR_MODE,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.color_mode = color_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = load_image(path, color_mode=self.color_mode)
        tensor = self.transform(image)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Transform pipeline must return a tensor")
        return tensor, label


def build_dataloaders(
    dataset_root: str | Path,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    color_mode: str = DEFAULT_COLOR_MODE,
    batch_size: int = 32,
    seed: int = DEFAULT_SEED,
    num_workers: int = 0,
    resplit: bool = False,
    split_file: str | Path | None = None,
) -> tuple[dict[str, DataLoader], DatasetSplit]:
    dataset_split = prepare_dataset(
        dataset_root,
        seed=seed,
        split_file=split_file,
        resplit=resplit,
    )
    train_transform = build_train_transforms(image_size)
    eval_transform = build_eval_transforms(image_size)

    loaders = {
        "train": DataLoader(
            ImageClassificationDataset(
                dataset_split.samples["train"],
                train_transform,
                color_mode=color_mode,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed),
        ),
        "val": DataLoader(
            ImageClassificationDataset(
                dataset_split.samples["val"],
                eval_transform,
                color_mode=color_mode,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            ImageClassificationDataset(
                dataset_split.samples["test"],
                eval_transform,
                color_mode=color_mode,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    return loaders, dataset_split
