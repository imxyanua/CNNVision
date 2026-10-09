import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.metrics import compute_classification_metrics
from evaluation.run import evaluate_model
from training.loop import train_model


def _fill_class(class_dir: Path, count: int, prefix: str) -> None:
    class_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        Image.new("RGB", (20, 16), color=(idx * 3, 40, 80)).save(class_dir / f"{prefix}_{idx:02d}.png")


class MetricsTest(unittest.TestCase):
    def test_known_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        metrics = compute_classification_metrics(y_true, y_pred, ["alpha", "beta"])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["num_samples"], 4)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertIn("precision_macro", metrics)
        self.assertIn("recall_macro", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("alpha", metrics["classification_report"])

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_classification_metrics(np.array([]), np.array([]), ["a", "b"])


class EvaluateModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmpdir.name)
        _fill_class(cls.root / "alpha", 8, "a")
        _fill_class(cls.root / "beta", 8, "b")
        cls.ckpt = cls.root / "best_model.pth"
        train_model(
            cls.root,
            output_path=cls.ckpt,
            epochs=1,
            batch_size=4,
            image_size=32,
            seed=0,
            device="cpu",
            num_workers=0,
            output_dir=cls.root / "train_outputs",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def setUp(self) -> None:
        self.out = self.root / "outputs" / self.id().split(".")[-1]

    def test_evaluate_writes_report_and_plot(self) -> None:
        metrics = evaluate_model(
            self.root,
            self.ckpt,
            output_dir=self.out,
            batch_size=4,
            device="cpu",
        )
        self.assertGreater(metrics["num_samples"], 0)
        self.assertTrue((self.out / "classification_report.txt").is_file())
        self.assertTrue((self.out / "metrics.json").is_file())
        self.assertTrue((self.out / "confusion_matrix.png").is_file())
        saved = json.loads((self.out / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["class_names"], ["alpha", "beta"])
        self.assertEqual(saved["num_samples"], metrics["num_samples"])

    def test_class_map_mismatch_raises(self) -> None:
        other = self.root / "other"
        _fill_class(other / "gamma", 8, "g")
        _fill_class(other / "delta", 8, "d")
        with self.assertRaises(ValueError):
            evaluate_model(other, self.ckpt, output_dir=self.out / "bad", device="cpu")

    def test_missing_checkpoint_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            evaluate_model(self.root, self.root / "missing.pth", output_dir=self.out, device="cpu")
