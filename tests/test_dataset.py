import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from training import build_dataloaders, prepare_dataset


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 18), color=(10, 20, 30)).save(path)


def _fill_class(class_dir: Path, count: int, prefix: str) -> None:
    for idx in range(count):
        _write_png(class_dir / f"{prefix}_{idx:02d}.png")


class DatasetSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _flat_dataset(self) -> Path:
        _fill_class(self.root / "alpha", 10, "a")
        _fill_class(self.root / "beta", 10, "b")
        return self.root

    def test_class_names_come_from_folders(self) -> None:
        split = prepare_dataset(self._flat_dataset(), seed=0)
        self.assertEqual(list(split.class_to_idx), ["alpha", "beta"])
        self.assertEqual(split.num_classes, 2)
        self.assertNotIn("cat", split.class_to_idx)
        self.assertNotIn("dog", split.class_to_idx)

    def test_splits_do_not_overlap(self) -> None:
        split = prepare_dataset(self._flat_dataset(), seed=0)
        train = {path.resolve() for path, _label in split.samples["train"]}
        val = {path.resolve() for path, _label in split.samples["val"]}
        test = {path.resolve() for path, _label in split.samples["test"]}
        self.assertTrue(train)
        self.assertTrue(val)
        self.assertTrue(test)
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)

    def test_same_seed_same_split(self) -> None:
        first = prepare_dataset(self._flat_dataset(), seed=3, resplit=True)
        second = prepare_dataset(self.root, seed=3, resplit=True)
        self.assertEqual(_path_names(first, "train"), _path_names(second, "train"))
        self.assertEqual(_path_names(first, "val"), _path_names(second, "val"))

    def test_saved_split_is_reused(self) -> None:
        first = prepare_dataset(self._flat_dataset(), seed=1)
        # A different seed must not reshuffle when split.json already exists.
        second = prepare_dataset(self.root, seed=99, resplit=False)
        self.assertEqual(_path_names(first, "train"), _path_names(second, "train"))
        self.assertEqual(first.seed, second.seed)

    def test_counts_cover_every_class(self) -> None:
        split = prepare_dataset(self._flat_dataset(), seed=0)
        counts = split.counts()
        for split_name in ("train", "val", "test"):
            self.assertEqual(set(counts[split_name]), {"alpha", "beta"})
            self.assertGreater(sum(counts[split_name].values()), 0)

    def test_presplit_folders_are_used(self) -> None:
        _fill_class(self.root / "train" / "alpha", 3, "tr_a")
        _fill_class(self.root / "train" / "beta", 3, "tr_b")
        _fill_class(self.root / "val" / "alpha", 2, "va_a")
        _fill_class(self.root / "val" / "beta", 2, "va_b")
        _fill_class(self.root / "test" / "alpha", 1, "te_a")
        _fill_class(self.root / "test" / "beta", 1, "te_b")
        split = prepare_dataset(self.root, seed=0)
        self.assertEqual(split.layout, "presplit")
        self.assertEqual(len(split.samples["train"]), 6)
        self.assertEqual(len(split.samples["val"]), 4)
        self.assertEqual(len(split.samples["test"]), 2)

    def test_empty_class_folder_raises(self) -> None:
        _fill_class(self.root / "alpha", 4, "a")
        (self.root / "beta").mkdir()
        with self.assertRaises(ValueError):
            prepare_dataset(self.root)

    def test_one_class_raises(self) -> None:
        _fill_class(self.root / "alpha", 8, "a")
        with self.assertRaises(ValueError):
            prepare_dataset(self.root)

    def test_split_json_lists_relative_paths(self) -> None:
        prepare_dataset(self._flat_dataset(), seed=0)
        payload = json.loads((self.root / "split.json").read_text(encoding="utf-8"))
        rel = payload["samples"]["train"][0][0]
        self.assertNotIn("\\", rel)
        self.assertTrue((self.root / rel).is_file())


class DataLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        _fill_class(self.root / "alpha", 8, "a")
        _fill_class(self.root / "beta", 8, "b")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_batch_shape_and_labels(self) -> None:
        loaders, split = build_dataloaders(self.root, image_size=32, batch_size=4, seed=0)
        images, labels = next(iter(loaders["train"]))
        self.assertEqual(tuple(images.shape), (4, 3, 32, 32))
        self.assertEqual(labels.dtype, torch.int64)
        self.assertTrue(set(labels.tolist()).issubset(set(split.idx_to_class)))

    def test_train_loader_uses_augment_eval_does_not(self) -> None:
        loaders, _split = build_dataloaders(self.root, image_size=32, batch_size=4, seed=0)
        train_ops = [type(op).__name__ for op in loaders["train"].dataset.transform.transforms]
        val_ops = [type(op).__name__ for op in loaders["val"].dataset.transform.transforms]
        test_ops = [type(op).__name__ for op in loaders["test"].dataset.transform.transforms]
        self.assertIn("RandomHorizontalFlip", train_ops)
        self.assertNotIn("RandomHorizontalFlip", val_ops)
        self.assertNotIn("RandomHorizontalFlip", test_ops)
        self.assertEqual(val_ops, test_ops)
        self.assertIsInstance(loaders["val"].dataset.transform, transforms.Compose)


def _path_names(split, split_name: str) -> list[str]:
    return sorted(path.name for path, _label in split.samples[split_name])
