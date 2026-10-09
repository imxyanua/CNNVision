import ast
import unittest
from pathlib import Path


class WebAppTest(unittest.TestCase):
    def test_app_calls_shared_predict(self) -> None:
        source = Path("app.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("from inference.predict import predict_image", source)
        self.assertIn("from app_i18n import LANG_LABELS, translate", source)
        self.assertIn("predict_image(", source)
        self.assertIn("probability_network_svg", source)
        self.assertNotIn("build_cnn(", source)
        self.assertNotIn("augment=True", source)
