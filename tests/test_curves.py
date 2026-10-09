import tempfile
import unittest
from pathlib import Path

from visualization.curves import save_training_curves


class TrainingCurvesTest(unittest.TestCase):
    def test_saves_png_from_history(self) -> None:
        history = [
            {"epoch": 1, "train_loss": 1.2, "val_loss": 1.1, "train_acc": 0.4, "val_acc": 0.5},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 1.0, "train_acc": 0.6, "val_acc": 0.55},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_curves.png"
            saved = save_training_curves(history, path)
            self.assertTrue(saved.is_file())
            self.assertGreater(saved.stat().st_size, 0)

    def test_empty_history_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_training_curves([], Path(tmp) / "training_curves.png")
