import unittest

import torch

from models import SimpleCNN, build_cnn, build_cnn_from_settings


class SimpleCNNTest(unittest.TestCase):
    def test_logits_shape_matches_num_classes(self) -> None:
        model = build_cnn(num_classes=4, in_channels=3)
        logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, 4))

    def test_grayscale_input(self) -> None:
        model = build_cnn(num_classes=3, in_channels=1)
        logits = model(torch.randn(2, 1, 48, 48))
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_gap_accepts_different_image_sizes(self) -> None:
        model = build_cnn(num_classes=2, in_channels=3)
        small = model(torch.randn(1, 3, 32, 32))
        large = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(small.shape), (1, 2))
        self.assertEqual(tuple(large.shape), (1, 2))

    def test_forward_returns_logits_not_softmax(self) -> None:
        model = build_cnn(num_classes=3, in_channels=3)
        model.eval()
        x = torch.randn(4, 3, 32, 32)
        logits = model(x)
        # Forward must stay as logits so CrossEntropyLoss does not get a second softmax.
        self.assertFalse(torch.allclose(logits, torch.softmax(logits, dim=1)))

    def test_probabilities_sum_to_one(self) -> None:
        model = build_cnn(num_classes=5, in_channels=3)
        model.eval()
        probs = model.probabilities(torch.randn(3, 3, 32, 32))
        self.assertEqual(tuple(probs.shape), (3, 5))
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5))
        self.assertTrue(torch.all(probs >= 0))

    def test_rebuild_from_settings(self) -> None:
        model = build_cnn(num_classes=6, in_channels=1)
        rebuilt = build_cnn_from_settings(model.settings())
        self.assertIsInstance(rebuilt, SimpleCNN)
        self.assertEqual(rebuilt.num_classes, 6)
        self.assertEqual(rebuilt.in_channels, 1)

    def test_rejects_single_class(self) -> None:
        with self.assertRaises(ValueError):
            build_cnn(num_classes=1)
