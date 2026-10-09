import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from models import load_model_from_checkpoint
from training.loop import train_model


def _fill_class(class_dir: Path, count: int, prefix: str) -> None:
    class_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        Image.new("RGB", (20, 16), color=(idx * 3, 40, 80)).save(class_dir / f"{prefix}_{idx:02d}.png")


class TrainLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        _fill_class(self.root / "alpha", 8, "a")
        _fill_class(self.root / "beta", 8, "b")
        self.ckpt = self.root / "best_model.pth"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_one_epoch_saves_loadable_checkpoint(self) -> None:
        result = train_model(
            self.root,
            output_path=self.ckpt,
            epochs=1,
            batch_size=4,
            image_size=32,
            seed=0,
            device="cpu",
            num_workers=0,
            output_dir=self.root / "outputs",
        )
        self.assertTrue(self.ckpt.is_file())
        self.assertTrue((self.root / "outputs" / "training_curves.png").is_file())
        self.assertTrue(result["curves_path"].is_file())
        self.assertEqual(result["num_classes"], 2)
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(set(result["class_to_idx"]), {"alpha", "beta"})

        model, payload = load_model_from_checkpoint(self.ckpt, map_location="cpu")
        self.assertEqual(model.num_classes, 2)
        self.assertEqual(payload["class_to_idx"], result["class_to_idx"])
        self.assertEqual(payload["preprocess"]["image_size"], 32)
        self.assertEqual(payload["preprocess"]["color_mode"], "rgb")
        self.assertEqual(payload["preprocess"]["normalize"], "to_tensor_0_1")
        self.assertIn("val_acc", payload)
        self.assertEqual(len(payload["history"]), 1)

    def test_checkpoint_round_trip_forward(self) -> None:
        train_model(
            self.root,
            output_path=self.ckpt,
            epochs=1,
            batch_size=4,
            image_size=32,
            seed=1,
            device="cpu",
            output_dir=self.root / "outputs",
        )
        model, payload = load_model_from_checkpoint(self.ckpt, map_location="cpu")
        model.eval()
        logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, payload["model"]["num_classes"]))

    def test_checkpoint_history_covers_every_epoch(self) -> None:
        train_model(
            self.root,
            output_path=self.ckpt,
            epochs=2,
            batch_size=4,
            image_size=32,
            seed=2,
            device="cpu",
            output_dir=self.root / "outputs",
        )
        _model, payload = load_model_from_checkpoint(self.ckpt, map_location="cpu")
        self.assertEqual(len(payload["history"]), 2)
        self.assertEqual([row["epoch"] for row in payload["history"]], [1, 2])
