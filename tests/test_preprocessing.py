"""CPU tests for image loading and shared train/eval transforms."""

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from preprocessing import (
    build_eval_transforms,
    build_train_transforms,
    load_image,
    preprocess_image,
    preprocess_settings,
)


def _write_rgb_image(path: Path, size: tuple[int, int] = (40, 32)) -> None:
    width, height = size
    image = Image.new("RGB", (width, height), color=(12, 80, 200))
    image.save(path)


class PreprocessingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.image_path = Path(self._tmpdir.name) / "sample.png"
        _write_rgb_image(self.image_path)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_eval_tensor_shape_and_range(self) -> None:
        tensor = preprocess_image(self.image_path, image_size=64, augment=False)
        self.assertEqual(tuple(tensor.shape), (3, 64, 64))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertGreaterEqual(tensor.min().item(), 0.0)
        self.assertLessEqual(tensor.max().item(), 1.0)

    def test_grayscale_has_one_channel(self) -> None:
        tensor = preprocess_image(
            self.image_path,
            image_size=32,
            color_mode="grayscale",
            augment=False,
        )
        self.assertEqual(tuple(tensor.shape), (1, 32, 32))

    def test_eval_is_deterministic(self) -> None:
        first = preprocess_image(self.image_path, image_size=48, augment=False)
        second = preprocess_image(self.image_path, image_size=48, augment=False)
        self.assertTrue(torch_equal(first, second))

    def test_train_transforms_include_augmentation(self) -> None:
        train_ops = type_names(build_train_transforms(64))
        eval_ops = type_names(build_eval_transforms(64))
        self.assertIn("RandomResizedCrop", train_ops)
        self.assertIn("RandomHorizontalFlip", train_ops)
        self.assertIn("RandomRotation", train_ops)
        self.assertIn("ColorJitter", train_ops)
        self.assertNotIn("RandomResizedCrop", eval_ops)
        self.assertNotIn("RandomHorizontalFlip", eval_ops)
        self.assertIn("Resize", eval_ops)
        self.assertEqual(train_ops[-1], "ToTensor")
        self.assertEqual(eval_ops[-1], "ToTensor")

    def test_missing_file_raises(self) -> None:
        missing = Path(self._tmpdir.name) / "missing.jpg"
        with self.assertRaises(FileNotFoundError):
            load_image(missing)

    def test_unsupported_file_raises(self) -> None:
        text_path = Path(self._tmpdir.name) / "notes.txt"
        text_path.write_text("not an image", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_image(text_path)

    def test_preprocess_settings_round_trip_fields(self) -> None:
        settings = preprocess_settings(image_size=96, color_mode="grayscale")
        self.assertEqual(settings["image_size"], 96)
        self.assertEqual(settings["color_mode"], "grayscale")
        self.assertEqual(settings["normalize"], "to_tensor_0_1")


def type_names(compose: transforms.Compose) -> list[str]:
    return [type(op).__name__ for op in compose.transforms]


def torch_equal(left, right) -> bool:
    return bool((left == right).all().item())


if __name__ == "__main__":
    unittest.main()
