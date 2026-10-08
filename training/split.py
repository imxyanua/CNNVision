from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from preprocessing.image import is_image_file

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
# Folder names that mean a ready-made split, not a class.
_PRESPLIT_DIRS = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}


@dataclass(frozen=True)
class DatasetSplit:
    root: Path
    layout: str
    seed: int
    class_to_idx: dict[str, int]
    samples: dict[str, list[tuple[Path, int]]]

    @property
    def idx_to_class(self) -> dict[int, str]:
        return {idx: name for name, idx in self.class_to_idx.items()}

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def counts(self) -> dict[str, dict[str, int]]:
        idx_to_class = self.idx_to_class
        result: dict[str, dict[str, int]] = {}
        for split_name, items in self.samples.items():
            per_class = {name: 0 for name in self.class_to_idx}
            for _path, label in items:
                per_class[idx_to_class[label]] += 1
            result[split_name] = per_class
        return result


def prepare_dataset(
    dataset_root: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    split_file: str | Path | None = None,
    resplit: bool = False,
) -> DatasetSplit:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Dataset path is not a directory: {root}")

    _check_ratios(train_ratio, val_ratio, test_ratio)
    split_path = Path(split_file) if split_file is not None else root / "split.json"

    # Reuse a saved file list so a later run does not shuffle again.
    if split_path.is_file() and not resplit:
        return _load_split(root, split_path)

    if _is_presplit(root):
        dataset_split = _from_presplit_folders(root, seed)
    else:
        dataset_split = _from_flat_folders(
            root,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    _validate_split(dataset_split)
    _save_split(dataset_split, split_path)
    return dataset_split


def _check_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be >= 0")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")


def _is_presplit(root: Path) -> bool:
    train_dir = root / "train"
    if not train_dir.is_dir():
        return False
    return any(path.is_dir() and not path.name.startswith(".") for path in train_dir.iterdir())


def _iter_class_dirs(root: Path) -> list[Path]:
    dirs = [
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    dirs.sort(key=lambda path: path.name)
    return dirs


def _iter_images(class_dir: Path) -> list[Path]:
    files = [path for path in class_dir.iterdir() if path.is_file() and is_image_file(path)]
    files.sort(key=lambda path: path.name)
    return files


def _class_to_idx(class_names: list[str]) -> dict[str, int]:
    if not class_names:
        raise ValueError("No class folders with images found")
    return {name: idx for idx, name in enumerate(class_names)}


def _from_flat_folders(
    root: Path,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> DatasetSplit:
    class_dirs = [path for path in _iter_class_dirs(root) if path.name not in _PRESPLIT_DIRS]
    class_images: dict[str, list[Path]] = {}
    for class_dir in class_dirs:
        images = _iter_images(class_dir)
        if not images:
            raise ValueError(f"Class folder has no readable images: {class_dir}")
        class_images[class_dir.name] = images

    class_to_idx = _class_to_idx(list(class_images))
    samples: dict[str, list[tuple[Path, int]]] = {name: [] for name in SPLIT_NAMES}

    # Split per class so every class can appear in train/val/test when there are enough files.
    for class_name, images in class_images.items():
        label = class_to_idx[class_name]
        grouped = _split_paths(images, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        for split_name, paths in grouped.items():
            samples[split_name].extend((path, label) for path in paths)

    return DatasetSplit(
        root=root.resolve(),
        layout="flat",
        seed=seed,
        class_to_idx=class_to_idx,
        samples=samples,
    )


def _from_presplit_folders(root: Path, seed: int) -> DatasetSplit:
    grouped: dict[str, dict[str, list[Path]]] = {name: {} for name in SPLIT_NAMES}
    for child in _iter_class_dirs(root):
        split_name = _PRESPLIT_DIRS.get(child.name)
        if split_name is None:
            continue
        for class_dir in _iter_class_dirs(child):
            images = _iter_images(class_dir)
            if not images:
                raise ValueError(f"Class folder has no readable images: {class_dir}")
            grouped[split_name][class_dir.name] = images

    if not grouped["train"]:
        raise ValueError(f"Pre-split dataset has no train classes: {root}")

    class_names = sorted(grouped["train"])
    extra = set().union(*[set(grouped[name]) for name in SPLIT_NAMES]) - set(class_names)
    if extra:
        raise ValueError(f"Classes {sorted(extra)} appear in val/test but not in train")

    class_to_idx = _class_to_idx(class_names)
    samples: dict[str, list[tuple[Path, int]]] = {name: [] for name in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        for class_name, paths in grouped[split_name].items():
            label = class_to_idx[class_name]
            samples[split_name].extend((path, label) for path in paths)

    return DatasetSplit(
        root=root.resolve(),
        layout="presplit",
        seed=seed,
        class_to_idx=class_to_idx,
        samples=samples,
    )


def _split_paths(
    paths: list[Path],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[Path]]:
    items = list(paths)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_train = int(len(items) * train_ratio)
    n_val = int(len(items) * val_ratio)
    # Leftover files go to test so every image is used once.
    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }


def _validate_split(dataset_split: DatasetSplit) -> None:
    seen: set[Path] = set()
    for split_name, items in dataset_split.samples.items():
        for path, _label in items:
            resolved = path.resolve()
            if resolved in seen:
                raise ValueError(f"File appears in more than one split: {resolved}")
            seen.add(resolved)
        if split_name == "train" and not items:
            raise ValueError("Training split is empty")

    if dataset_split.num_classes < 2:
        raise ValueError("Need at least two classes for multi-class classification")


def _save_split(dataset_split: DatasetSplit, split_path: Path) -> None:
    payload = {
        "root": str(dataset_split.root),
        "layout": dataset_split.layout,
        "seed": dataset_split.seed,
        "class_to_idx": dataset_split.class_to_idx,
        "samples": {
            split_name: [
                [_relative(path, dataset_split.root), label]
                for path, label in items
            ]
            for split_name, items in dataset_split.samples.items()
        },
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_split(root: Path, split_path: Path) -> DatasetSplit:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    class_to_idx = {str(name): int(idx) for name, idx in payload["class_to_idx"].items()}
    samples: dict[str, list[tuple[Path, int]]] = {name: [] for name in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        for rel, label in payload["samples"].get(split_name, []):
            path = (root / rel).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Split lists {rel} but the file is missing: {path}")
            samples[split_name].append((path, int(label)))

    dataset_split = DatasetSplit(
        root=root.resolve(),
        layout=str(payload["layout"]),
        seed=int(payload["seed"]),
        class_to_idx=class_to_idx,
        samples=samples,
    )
    _validate_split(dataset_split)
    return dataset_split


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
