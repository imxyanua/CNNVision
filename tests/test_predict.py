import tempfile
import unittest
from pathlib import Path

from PIL import Image

from inference import predict_image
from training.loop import train_model


def _fill_class(class_dir: Path, count: int, prefix: str) -> None:
    class_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        Image.new("RGB", (20, 16), color=(idx * 3, 40, 80)).save(class_dir / f"{prefix}_{idx:02d}.png")


class PredictImageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmpdir.name)
        _fill_class(cls.root / "alpha", 8, "a")
        _fill_class(cls.root / "beta", 8, "b")
        cls.ckpt = cls.root / "best_model.pth"
        cls.image = cls.root / "query.png"
        Image.new("RGB", (28, 22), color=(12, 80, 200)).save(cls.image)
        train_model(
            cls.root,
            output_path=cls.ckpt,
            epochs=1,
            batch_size=4,
            image_size=32,
            seed=0,
            device="cpu",
            num_workers=0,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_prediction_uses_class_names_and_probabilities(self) -> None:
        result = predict_image(self.image, self.ckpt, device="cpu")
        self.assertIn(result["class_name"], {"alpha", "beta"})
        self.assertIn(result["class_index"], {0, 1})
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)
        self.assertEqual(set(result["probabilities"]), {"alpha", "beta"})
        self.assertEqual(result["probabilities"][result["class_name"]], result["confidence"])

    def test_missing_image_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            predict_image(self.root / "missing.jpg", self.ckpt, device="cpu")

    def test_missing_checkpoint_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            predict_image(self.image, self.root / "missing.pth", device="cpu")

    def test_unsupported_file_raises(self) -> None:
        text_path = self.root / "notes.txt"
        text_path.write_text("not an image", encoding="utf-8")
        with self.assertRaises(ValueError):
            predict_image(text_path, self.ckpt, device="cpu")
